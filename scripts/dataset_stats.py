"""Task 2: MVTec AD dataset analysis for the selected categories.

Usage:
    MVTEC_ROOT=... python scripts/dataset_stats.py [--root DIR] [--out reports/dataset]
                                                   [--categories bottle tile ...]

For each category it reports:
- the number of train/good images, and how our fixed-seed split divides them into fit and val
  (``ad.data.split_train_val``),
- test counts per defect type,
- the original image resolution(s),
- the defect area as a fraction of mask pixels per defect image (mean / median / min / max),
- the number of 8-connected defect regions per defect image,
- two cheap proxies for how much normal images vary, computed on train/good only:
  ``pix_std`` is the mean per-pixel std across 64x64 grey images. ``align_corr`` is the mean
  Pearson correlation of each normal image with the mean normal image. It is high for aligned
  objects on a fixed background, and low for rotated objects or stochastic textures.

Outputs in --out: dataset_stats.md, dataset_stats.csv (one row per category x defect type),
defect_regions_per_image.csv, samples_<cat>.png and defect_area_hist.png.
GT masks are read only for descriptive statistics. Nothing here touches model fitting or thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ad.data import (CATEGORIES, IMG_EXTS, OBJECT_CATEGORIES, get_root,  # noqa: E402
                     split_train_val)

KIND_COLOR = {"object": "#2a78d6", "texture": "#eb6834"}  # categorical slots 1-2 (dataviz palette)
N_VAR_IMAGES = 60  # normal images used for the variation proxies

# Qualitative notes per category, from inspecting the images and from the MVTec AD papers.
# {placeholders} are filled with the measured statistics.
DISCUSSION = {
    "bottle": (
        "**bottle (object).** Bottles are photographed top-down, looking into the opening. The "
        "bottle fills most of the frame, is centred on a white background and shows no rotation "
        "of note (align_corr = {align_corr:.3f}). Because of the "
        "radial symmetry, normal appearance at a given image position is almost the same in every "
        "image. This favours per-position Gaussian models like PaDiM, and also suits PatchCore's "
        "memory bank. Defects (broken_large, broken_small, contamination) are mid-sized: the median "
        "defect area is {area_med:.1%} of the image, with {reg_mean:.1f} region(s) per image on "
        "average. They are easy to detect at image level. The hard part is localisation along the "
        "glossy rim, where specular highlights can raise false positives."
    ),
    "capsule": (
        "**capsule (object).** A single pill lies horizontally on a plain light background, in the "
        "same orientation every time: black cap on the left, '500' imprint on the right. Normal "
        "images are as well aligned as bottle (align_corr = {align_corr:.3f}), so per-position "
        "models such as PaDiM apply directly. Defects are small and subtle: cracks, pokes, "
        "scratches, faulty imprints and squeeze. The median defect area is only {area_med:.2%} of "
        "the image (min {area_min:.3%}). On the 28x28 stride-8 feature grid at 224x224 input, that "
        "is about 3 of 784 cells, and the smallest cracks cover less than one cell. So capsule "
        "tests the spatial resolution of the features, and Gaussian smoothing can wash small "
        "defects out. Squeeze is the exception: a large shape change (median 2.8%) that only shows "
        "as a slightly deformed outline. The imprint and logo are normal high-frequency structure, "
        "and a faulty_imprint differs from a correct one only in fine detail."
    ),
    "screw": (
        "**screw (object).** The screw lies at an arbitrary in-plane rotation and position on a "
        "light grey background with a mild illumination gradient. Its normal images are by far the "
        "least aligned of the objects (align_corr = {align_corr:.2f} vs about 1.00 for bottle and "
        "capsule; pix_std = {pix_std:.3f}). This breaks PaDiM's assumption that the same "
        "position looks the same in every image, because each position sees thread, head or "
        "background depending on the rotation. PatchCore's position-agnostic nearest-neighbour "
        "memory copes better. Defects such as manipulated_front, scratch_head, scratch_neck, "
        "thread_side and thread_top are uniformly tiny: median {area_med:.2%}, max {area_max:.2%}, "
        "the narrowest size range of the six. thread_side defects are often split into several "
        "small regions. Screw is reported in the PaDiM and PatchCore papers as one of the hardest "
        "MVTec AD categories, and we expect it to be the hardest of the six here."
    ),
    "carpet": (
        "**carpet (texture).** A woven, stationary texture fills the whole frame. There is no "
        "object geometry or background, and alignment is irrelevant (align_corr = {align_corr:.2f} "
        "only shows that the texture is stochastic). Every patch is statistically alike, so a "
        "memory of normal patches generalises well and PatchCore can sub-sample it aggressively. "
        "Defects (color, cut, hole, metal_contamination, thread) vary in size, with a median of "
        "{area_med:.2%}. cut defects are the largest (median about 4%), while "
        "metal_contamination is the smallest (median about 0.4%) and is easy to miss after "
        "downsampling. color defects keep the weave structure intact and change only the hue, "
        "so they rely on how well the backbone features encode colour."
    ),
    "tile": (
        "**tile (texture).** Ceramic tile with irregular dark-grey blotches on a light grey-green base. Like carpet it is "
        "stationary with no meaningful alignment (align_corr = {align_corr:.2f}). Its normal images "
        "also vary the most of all six categories (pix_std = {pix_std:.3f}), so the normal "
        "distribution is broad and normal speckle clusters can produce false positives. Defects (crack, glue_strip, gray_stroke, oil, "
        "rough) are the largest in this study: median {area_med:.1%}, max {area_max:.0%}. That "
        "makes image-level detection easy. Localisation must still capture the whole extent of "
        "oil and glue stains, which are low-contrast at their borders and pull pixel AUPRO down."
    ),
    "leather": (
        "**leather (texture).** Fine, uniform grain with no object geometry. Its align_corr is "
        "high for a texture ({align_corr:.2f}) because every image shares the same smooth "
        "illumination fall-off towards the corners, not because the grain is aligned. Pixel "
        "variation between normal images is the lowest of the textures (pix_std = {pix_std:.3f}), "
        "so most defects (color, cut, fold, glue, poke) stand out clearly against the grain. "
        "Defects are small (median {area_med:.2%}, min {area_min:.3%}). Pokes and colour spots are "
        "the smallest, with medians of about 0.2%. Leather is usually close to "
        "saturated for pretrained-feature methods. The main failure mode is a slight colour "
        "change, which mid-level CNN features are less sensitive to than to structural changes."
    ),
}


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.is_dir() else []


def _mask_path(cat_dir: Path, defect: str, img: Path) -> Path:
    return cat_dir / "ground_truth" / defect / f"{img.stem}_mask.png"


def _variation_proxies(paths: list[Path]) -> tuple[float, float]:
    if not paths:
        return float("nan"), float("nan")
    idx = np.linspace(0, len(paths) - 1, min(N_VAR_IMAGES, len(paths))).round().astype(int)
    stack = np.stack([
        cv2.resize(np.asarray(Image.open(paths[i]).convert("L"), np.float32) / 255.0, (64, 64),
                   interpolation=cv2.INTER_AREA).ravel() for i in sorted(set(idx.tolist()))])
    pix_std = float(stack.std(axis=0).mean())
    mean = stack.mean(axis=0)
    corr = [np.corrcoef(x, mean)[0, 1] if x.std() > 0 and mean.std() > 0 else np.nan for x in stack]
    return pix_std, float(np.nanmean(corr))


def analyse_category(cat_dir: Path, val_ratio: float = 0.1, seed: int = 0):
    """Return (summary dict, per-defect-type rows, per-image rows) for one category folder."""
    cat = cat_dir.name
    train = _images(cat_dir / "train" / "good")
    fit, val = split_train_val(train, val_ratio, seed)
    sizes: set[tuple[int, int]] = set()
    for p in train[:5]:
        sizes.add(Image.open(p).size)

    type_rows, img_rows = [], []
    test_root = cat_dir / "test"
    defect_types = sorted(d.name for d in test_root.iterdir() if d.is_dir()) if test_root.is_dir() else []
    for dt in defect_types:
        imgs = _images(test_root / dt)
        for p in imgs[:3]:
            sizes.add(Image.open(p).size)
        areas, regions = [], []
        if dt != "good":
            for p in imgs:
                mp = _mask_path(cat_dir, dt, p)
                if not mp.is_file():
                    continue
                m = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.uint8)
                n_reg = cv2.connectedComponents(m, connectivity=8)[0] - 1
                areas.append(float(m.mean()))
                regions.append(int(n_reg))
                img_rows.append({"category": cat, "defect_type": dt, "file": p.name,
                                 "height": m.shape[0], "width": m.shape[1],
                                 "area_frac": areas[-1], "n_regions": n_reg})
        a, r = np.array(areas), np.array(regions)
        type_rows.append({
            "category": cat, "defect_type": dt, "n_test": len(imgs), "n_masks": len(areas),
            "area_mean": a.mean() if a.size else np.nan, "area_median": np.median(a) if a.size else np.nan,
            "area_min": a.min() if a.size else np.nan, "area_max": a.max() if a.size else np.nan,
            "regions_mean": r.mean() if r.size else np.nan, "regions_max": r.max() if r.size else np.nan,
        })

    all_a = np.array([x["area_frac"] for x in img_rows])
    all_r = np.array([x["n_regions"] for x in img_rows])
    pix_std, align_corr = _variation_proxies(train)
    n_test = sum(t["n_test"] for t in type_rows)
    n_good = sum(t["n_test"] for t in type_rows if t["defect_type"] == "good")
    summary = {
        "category": cat, "kind": "object" if cat in OBJECT_CATEGORIES else "texture",
        "n_train_good": len(train), "n_fit": len(fit), "n_val": len(val),
        "n_test": n_test, "n_test_good": n_good, "n_test_defect": n_test - n_good,
        "n_defect_types": sum(t["defect_type"] != "good" for t in type_rows),
        "resolution": ", ".join(f"{w}x{h}" for w, h in sorted(sizes)),
        "area_mean": all_a.mean() if all_a.size else np.nan,
        "area_med": float(np.median(all_a)) if all_a.size else np.nan,
        "area_min": all_a.min() if all_a.size else np.nan,
        "area_max": all_a.max() if all_a.size else np.nan,
        "reg_mean": all_r.mean() if all_r.size else np.nan,
        "reg_max": int(all_r.max()) if all_r.size else 0,
        "multi_region_frac": float((all_r > 1).mean()) if all_r.size else np.nan,
        "pix_std": pix_std, "align_corr": align_corr,
    }
    return summary, type_rows, img_rows


# ------------------------------------------------------------------------------------- figures
def _fig(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def samples_figure(cat_dir: Path, out: Path, disp: int = 256) -> None:
    """One normal test image + the first image of every defect type, with the GT mask overlaid."""
    cat = cat_dir.name
    items = []
    goods = _images(cat_dir / "test" / "good") or _images(cat_dir / "train" / "good")
    if goods:
        items.append(("good", goods[0], None))
    for d in sorted(x for x in (cat_dir / "test").iterdir() if x.is_dir() and x.name != "good"):
        imgs = _images(d)
        if imgs:
            items.append((d.name, imgs[0], _mask_path(cat_dir, d.name, imgs[0])))
    if not items:
        return
    fig = _fig(2.3 * len(items), 2.7)
    axes = np.atleast_1d(fig.subplots(1, len(items)))
    for ax, (name, ip, mp) in zip(axes, items):
        img = np.asarray(Image.open(ip).convert("RGB").resize((disp, disp), Image.BILINEAR))
        ax.imshow(img)
        if mp is not None and mp.is_file():
            m = np.asarray(Image.open(mp).convert("L").resize((disp, disp), Image.NEAREST)) > 127
            rgba = np.zeros((disp, disp, 4), np.float32)
            rgba[m] = (1.0, 0.0, 0.0, 0.35)
            ax.imshow(rgba)
            if m.any():
                ax.contour(m.astype(np.float32), levels=[0.5], colors="red", linewidths=1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{cat}: normal vs one example per defect type (GT mask in red)", fontsize=10)
    fig.savefig(out, dpi=110)  # keeps the committed sample sheets small


def area_hist_figure(img_df: pd.DataFrame, summaries: list[dict], out: Path) -> None:
    """Small multiples (one per category), shared log-x bins, per-image defect area fraction."""
    cats = [s["category"] for s in summaries]
    kinds = {s["category"]: s["kind"] for s in summaries}
    a = img_df["area_frac"].to_numpy()
    a = a[a > 0]
    if a.size == 0:
        return
    bins = np.logspace(np.floor(np.log10(a.min())), np.ceil(np.log10(a.max())), 31)
    ncols = min(3, len(cats))
    nrows = int(np.ceil(len(cats) / ncols))
    fig = _fig(3.6 * ncols, 2.6 * nrows)
    axes = np.atleast_1d(fig.subplots(nrows, ncols, sharex=True, squeeze=False)).ravel()
    for ax, cat in zip(axes, cats):
        v = img_df.loc[(img_df["category"] == cat) & (img_df["area_frac"] > 0), "area_frac"].to_numpy()
        ax.hist(v, bins=bins, color=KIND_COLOR[kinds[cat]], edgecolor="white", linewidth=0.8)
        if v.size:
            med = float(np.median(v))
            ax.axvline(med, color="#333333", lw=1.0, ls="--")
            ax.text(med, ax.get_ylim()[1] * 0.92, f" median {med:.2%}", fontsize=8, color="#333333")
        ax.set_xscale("log")
        ax.set_title(f"{cat} ({kinds[cat]}, n={v.size})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(cats):]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("defect area (fraction of image pixels, log scale)", fontsize=8)
    for ax in axes[::ncols]:
        ax.set_ylabel("# defect images", fontsize=8)
    fig.suptitle("MVTec AD: GT defect area per anomalous test image "
                 "(blue = object, orange = texture)", fontsize=10)
    fig.savefig(out, dpi=150)


# ------------------------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _pct(x: float, nd: int = 2) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.{nd}f}%"


def write_markdown(summaries: list[dict], type_df: pd.DataFrame, out: Path, root: Path) -> None:
    s = pd.DataFrame(summaries)
    overview = pd.DataFrame({
        "category": s["category"], "kind": s["kind"], "resolution": s["resolution"],
        "train/good": s["n_train_good"], "fit / val": s["n_fit"].astype(str) + " / " + s["n_val"].astype(str),
        "test good": s["n_test_good"], "test defect": s["n_test_defect"],
        "defect types": s["n_defect_types"],
        "pix_std": s["pix_std"].map(lambda x: f"{x:.3f}"),
        "align_corr": s["align_corr"].map(lambda x: f"{x:.3f}"),
    })
    area = pd.DataFrame({
        "category": s["category"],
        "area mean": s["area_mean"].map(_pct), "area median": s["area_med"].map(_pct),
        "area min": s["area_min"].map(lambda x: _pct(x, 3)), "area max": s["area_max"].map(_pct),
        "regions/img mean": s["reg_mean"].map(lambda x: f"{x:.2f}"),
        "regions/img max": s["reg_max"],
        "imgs with >1 region": s["multi_region_frac"].map(lambda x: _pct(x, 0)),
    })
    per_type = type_df.copy()
    per_type = pd.DataFrame({
        "category": per_type["category"], "defect type": per_type["defect_type"],
        "n test": per_type["n_test"],
        "area median": per_type["area_median"].map(_pct),
        "area min": per_type["area_min"].map(lambda x: _pct(x, 3)),
        "area max": per_type["area_max"].map(_pct),
        "regions/img mean": per_type["regions_mean"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}"),
    })

    parts = [
        "# MVTec AD: dataset analysis (Task 2)",
        "",
        "Generated by `scripts/dataset_stats.py`. Data: MVTec AD (Bergmann et al., CVPR 2019; "
        "IJCV 2021), licence CC BY-NC-SA 4.0. "
        f"Categories: {', '.join(s['category'])}.",
        "",
        "## Overview",
        "",
        "`fit / val` is our fixed-seed split of train/good: 90% to fit the model, 10% as the "
        "normal-only validation set for thresholds (`ad.data.split_train_val`, seed 0). "
        "`pix_std` is the mean per-pixel std over 64x64 grey normal images. "
        "`align_corr` is the mean correlation of a normal image with the mean normal image. "
        "For objects, high means aligned images with fixed geometry, and low means pose or "
        "rotation variation. For textures it mostly reflects shared low-frequency illumination, "
        "because the stochastic texture itself is uncorrelated. Both proxies use train/good only.",
        "",
        _md_table(overview),
        "",
        "## Defect size and number of regions (anomalous test images, GT masks at original resolution)",
        "",
        "Area is the fraction of image pixels inside the GT mask. Regions are 8-connected "
        "components of the mask. The full distribution is in `defect_area_hist.png`.",
        "",
        _md_table(area),
        "",
        "## Test images per defect type",
        "",
        _md_table(per_type),
        "",
        "![defect area histogram](defect_area_hist.png)",
        "",
        "## Category characteristics and expected impact on anomaly detection",
        "",
    ]
    for sm in summaries:
        txt = DISCUSSION.get(sm["category"])
        if txt:
            parts += [txt.format(**sm), "", f"![{sm['category']} samples](samples_{sm['category']}.png)", ""]
    parts += [
        "### Summary",
        "",
        "- **Texture vs object.** Texture categories (carpet, tile, leather) are stationary. Any "
        "patch can be compared with any normal patch, so position-agnostic methods (PatchCore) "
        "and position-specific ones (PaDiM) both work, and alignment does not matter. Object "
        "categories need consistent geometry for PaDiM's per-position statistics. That holds for "
        "bottle and capsule, which are both consistently posed, but not for screw, where rotation "
        "and position are free.",
        "- **Background variation.** The three object categories are shot on a controlled, uniform "
        "background, and the textures fill the frame with no background at all. For objects, most "
        "of the image is background, so the decision rests on a small foreground area, and "
        "background noise (dust, illumination gradients on screw) can cause false positives. "
        "Normal variation is highest for tile and screw (pix_std about 0.1), which makes the "
        "learned normal distribution broader and subtle defects harder to separate.",
        "- **Defect size.** Defect area spans about three orders of magnitude, from about 0.04% "
        "of the image (smallest capsule crack; leather and screw also go below 0.1%) to 35% "
        "(tile glue_strip). Screw, capsule and leather have medians below 0.5%, whereas tile and "
        "bottle defects are mostly above 2%. At 224x224 input, one "
        "stride-8 WRN50 feature cell covers about 0.13% of the image. Defects near or below that "
        "size are diluted by bilinear upsampling and Gaussian smoothing, which lowers both "
        "image-level recall and pixel-level overlap. Large defects are easy to detect, but they "
        "are harder to segment completely.",
        "- **Multiple regions.** Some images contain several separate defect regions (see the "
        "regions columns). AUPRO weights every region equally, so it rewards finding small "
        "secondary regions that pixel AUROC barely notices.",
        "",
    ]
    out.write_text("\n".join(parts))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="MVTec AD root (default: $MVTEC_ROOT)")
    ap.add_argument("--out", default=str(REPO / "reports" / "dataset"))
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    args = ap.parse_args(argv)

    root = get_root(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, type_rows, img_rows = [], [], []
    for cat in args.categories:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            print(f"[skip] {cat}: {cat_dir} not found", file=sys.stderr)
            continue
        sm, tr, ir = analyse_category(cat_dir)
        summaries.append(sm)
        type_rows += tr
        img_rows += ir
        samples_figure(cat_dir, out / f"samples_{cat}.png")
        print(f"[ok  ] {cat}: {sm['n_train_good']} train, {sm['n_test']} test, "
              f"{sm['n_defect_types']} defect types, median area {sm['area_med']:.4f}")
    if not summaries:
        raise SystemExit(f"no categories found under {root}")

    type_df = pd.DataFrame(type_rows)
    img_df = pd.DataFrame(img_rows, columns=["category", "defect_type", "file", "height", "width",
                                             "area_frac", "n_regions"])
    type_df.to_csv(out / "dataset_stats.csv", index=False, float_format="%.6g")
    img_df.to_csv(out / "defect_regions_per_image.csv", index=False, float_format="%.6g")
    area_hist_figure(img_df, summaries, out / "defect_area_hist.png")
    write_markdown(summaries, type_df, out / "dataset_stats.md", root)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
