"""Curated failure-case overlay panels: PaDiM vs PatchCore on the same test image.

Read-only, CPU only. Reads ``results/<run>/<category>/{scores.npz,maps.npz,metrics.json}``
of the ``padim_wrn50`` and ``patchcore_wrn50`` runs and writes one PNG per curated case to
``reports/figures/failure_cases/`` plus ``README.md`` with the captions.

Each panel has three columns:
  1. the 224x224 evaluation crop with the GT contour (green);
  2. PaDiM anomaly map, shown as map / pixel threshold, with the predicted region
     (map >= val pixel threshold) in red and GT in green;
  3. the same for PatchCore.
The column titles give the image score relative to each method's normal-validation image
threshold (ratio >= 1 means "anomalous") and the resulting decision (TP/FN/FP/TN).
Both heatmaps share one colour scale in threshold units (1.0 = pixel threshold), so the
colours are comparable across methods.

Usage::

    PYTHONPATH=src python scripts/analysis_failure_panels.py [--cases all|<id> ...]
    PYTHONPATH=src python scripts/analysis_failure_panels.py --candidates OUT_DIR \
        --category capsule --files scratch/017.png good/021.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ad.visualize import _load_image  # same resize -> centre-crop geometry as the dataset

RESULTS = Path("/home/eva/code/ma6216/industrial-visual-anomaly-detection/results")
RUNS = (("padim_wrn50", "PaDiM"), ("patchcore_wrn50", "PatchCore"))
OUT = Path("reports/figures/failure_cases")
VMIN, VMAX = 0.4, 1.6   # colour range in units of the pixel threshold

# id, category, "<defect_type>/<file>", caption (numbers are rendered from data in the titles).
CASES = [
    ("01", "screw", "thread_top/004.png",
     "**Missed only by PaDiM (pose).** The screw lies at an arbitrary rotation, so each PaDiM "
     "position Gaussian has to cover thread, shank, head and background seen across the fit "
     "images. The damaged thread tip does not stand out against that broad per-position "
     "distribution (PaDiM map stays below its pixel threshold everywhere). PatchCore's "
     "position-free nearest-neighbour search finds no normal patch like the damaged tip and "
     "localises it (pixel IoU 0.45)."),
    ("02", "screw", "thread_side/004.png",
     "**Missed by both.** A thread_side defect split into two tiny GT regions (0.14% of the full "
     "image) on the thread flank. The thread is high-frequency normal structure; at stride 8 "
     "the damaged flank still resembles normal thread patches, so neither map exceeds its pixel "
     "threshold inside the GT and both image scores stay below threshold."),
    ("03", "capsule", "scratch/017.png",
     "**Missed by both.** A faint, thin scratch next to the cap/body seam. PatchCore's map is "
     "flat (max 0.87x its pixel threshold). PaDiM's only above-threshold blob is a dust speck on "
     "the background (bottom right), not the scratch: its tight per-position background "
     "Gaussian reacts more to dust than to low-contrast damage on the body."),
    ("04", "capsule", "crack/016.png",
     "**Missed by both, peak off-target.** The crack is a thin sliver (0.26% of the crop). Both "
     "maps place their maxima on dark specks (one on the black cap, one on the background), "
     "which are normal-but-rare appearances, while the crack itself stays below threshold."),
    ("05", "capsule", "faulty_imprint/019.png",
     "**PatchCore near-miss: localised but not detected.** PatchCore's map exceeds its pixel "
     "threshold exactly on the damaged '5' (pixel IoU 0.45), but its Eq. 7 image score is only "
     "0.83x the image threshold. Pixel and image thresholds are calibrated separately on normal "
     "val data, and capsule has PatchCore's widest normal val score spread (image threshold "
     "1.58x the val median), so a pixel-level hit need not cross the image threshold. "
     "PaDiM's strongest response is again a background speck."),
    ("06", "tile", "gray_stroke/012.png",
     "**Missed by both: defect cut away by preprocessing.** The resize-256 / centre-crop-224 "
     "step keeps only ~10% of this gray_stroke's GT; what remains is a sliver at the bottom "
     "edge. The two lowest-scoring PatchCore tile misses (gray_stroke 012 and 002) are both "
     "crop-truncated. The tile maps also show how close normal dark blotches score to the "
     "threshold."),
    ("07", "tile", "oil/000.png",
     "**PaDiM threshold failure, not a ranking failure.** PaDiM's map clearly highlights the oil "
     "stain (map max 1.55x the pixel threshold, peak inside GT), yet the image score is 0.98x "
     "PaDiM's image threshold. That threshold (29.8) is driven by one extreme normal val image "
     "(val max 31.3 vs second-largest 24.4). PatchCore detects and segments it (IoU 0.85)."),
    ("08", "carpet", "good/012.png",
     "**False positive from a global shift (both methods, mostly PatchCore).** No local hotspot: "
     "PatchCore's map is raised almost everywhere and 42% of pixels exceed its pixel threshold. "
     "Normal test carpet scores sit above the normal val scores (median 1.112 vs 1.061), while "
     "PatchCore's val spread on carpet is very tight, so the 99% val quantile leaves little "
     "margin. 10 of 28 normal carpets are flagged."),
    ("09", "leather", "good/024.png",
     "**PatchCore false positive on normal leather.** No visible defect. The map is mildly raised "
     "with scattered above-threshold blobs, the largest at the left border. As on carpet, normal "
     "test leather scores higher than normal val (PatchCore median 1.070 vs 0.968) and the val "
     "spread is tight; PaDiM sees a similar shift but its looser threshold absorbs it."),
    ("10", "capsule", "good/006.png",
     "**PaDiM-only false positive: dust on the background.** A small dark speck above the "
     "capsule. PaDiM models each background position with a very tight Gaussian, so any speck "
     "is a large Mahalanobis outlier (score 1.23x threshold). PatchCore also responds weakly "
     "but stays below its image threshold."),
    ("11", "bottle", "good/015.png",
     "**PaDiM-only false positive on the rim.** A small bright spot on the upper-left of the "
     "glossy rim exceeds PaDiM's tight bottle threshold. The dataset analysis anticipated "
     "this: the rim's specular structure is where bottle false alarms arise. PatchCore stays "
     "below threshold."),
    ("12", "leather", "color/009.png",
     "**True positive with poor localisation.** Both methods detect the image and peak on the "
     "defect, but the thin colour streak becomes a blob several times wider after the "
     "stride-8 feature grid, bilinear upsampling and sigma=4 smoothing: pixel IoU is 0.07 for "
     "both. This is why leather has the lowest median TP IoU despite perfect image recall."),
]


def _load(run: str, category: str):
    rdir = RESULTS / run / category
    sc = np.load(rdir / "scores.npz", allow_pickle=False)
    mp = np.load(rdir / "maps.npz", allow_pickle=False)
    meta = json.loads((rdir / "metrics.json").read_text())
    keys = [f"{d}/{Path(p).name}" for d, p in zip(sc["defect_types"].astype(str), sc["paths"].astype(str))]
    return {"keys": keys, "scores": sc["image_scores"], "labels": sc["labels"],
            "paths": sc["paths"].astype(str), "maps": mp["maps"], "masks": mp["masks"],
            "image_thr": float(meta["image_threshold"]), "pixel_thr": float(meta["pixel_threshold"]),
            "data": meta["config"]["data"]}


def _decision(label: int, detected: bool) -> str:
    return {(1, True): "TP", (1, False): "FN", (0, True): "FP", (0, False): "TN"}[(label, detected)]


def _contour(ax, mask, color, lw):
    m = np.asarray(mask) > 0
    if m.any() and not m.all():
        ax.contour(m.astype(np.float32), levels=[0.5], colors=color, linewidths=lw)


def render(category: str, key: str, out_path: Path, title: str = "", cache: dict | None = None) -> dict:
    cache = {} if cache is None else cache
    runs = [cache.setdefault((run, category), _load(run, category)) for run, _ in RUNS]
    idx = runs[0]["keys"].index(key)
    if runs[1]["keys"][idx] != key:
        raise ValueError("test order differs between runs")
    data = runs[0]["data"]
    image = _load_image(runs[0]["paths"][idx], int(data["resize"]), int(data["crop"]),
                        runs[0]["maps"].shape[1:])
    gt = runs[0]["masks"][idx]
    label = int(runs[0]["labels"][idx])

    fig = Figure(figsize=(10.2, 3.6), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    axes[0].imshow(image)
    _contour(axes[0], gt, "lime", 1.2)
    axes[0].set_title(f"{category} / {key}\nGT (green)" if label else f"{category} / {key}\nnormal",
                      fontsize=9)
    info = {"category": category, "key": key, "label": label}
    for ax, r, (_, name) in zip(axes[1:], runs, RUNS):
        amap = r["maps"][idx].astype(np.float32)
        ratio_map = amap / r["pixel_thr"]
        score_ratio = float(r["scores"][idx]) / r["image_thr"]
        dec = _decision(label, score_ratio >= 1)
        ax.imshow(image)
        ax.imshow(ratio_map, cmap="jet", alpha=0.5, vmin=VMIN, vmax=VMAX)
        _contour(ax, ratio_map >= 1.0, "red", 2.0)
        _contour(ax, gt, "lime", 1.0)
        ax.set_title(f"{name}: {dec}  score/thr = {score_ratio:.2f}\n"
                     f"map max/pixel thr = {ratio_map.max():.2f}", fontsize=9)
        info[name] = {"decision": dec, "score_ratio": score_ratio, "map_max_ratio": float(ratio_map.max())}
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ScalarMappable(norm=Normalize(VMIN, VMAX), cmap="jet"), ax=axes[1:].tolist(),
                 fraction=0.03, pad=0.02, label="map / val pixel threshold")
    if title:
        fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    fig.clear()
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="*", default=["all"])
    ap.add_argument("--candidates", type=Path, help="render ad-hoc keys into this directory")
    ap.add_argument("--category")
    ap.add_argument("--files", nargs="*", default=[])
    args = ap.parse_args(argv)
    cache: dict = {}

    if args.candidates:
        for key in args.files:
            out = args.candidates / f"{args.category}_{key.replace('/', '_')}"
            print(render(args.category, key, out, cache=cache))
        return

    chosen = [c for c in CASES if "all" in args.cases or c[0] in args.cases]
    lines = ["# Curated failure cases (PaDiM vs PatchCore)", "",
             "Generated by `scripts/analysis_failure_panels.py` from the saved `padim_wrn50` and "
             "`patchcore_wrn50` outputs. Columns: image + GT (green); PaDiM map; PatchCore map. Maps "
             "are shown as map / normal-validation pixel threshold (shared colour scale "
             f"{VMIN}–{VMAX}); red contours = predicted region (map >= pixel threshold). Titles give "
             "image score / normal-validation image threshold and the decision. Per-image numbers "
             "are in `tables/per_image.csv`.", ""]
    for cid, category, key, caption in chosen:
        name = f"{cid}_{category}_{key.replace('/', '_')}"
        info = render(category, key, OUT / name, cache=cache)
        p, c = info["PaDiM"], info["PatchCore"]
        lines += [f"## {cid}. {category} / {key}", "", f"![{category} {key}]({name})", "",
                  f"PaDiM {p['decision']} (score/thr {p['score_ratio']:.2f}, map max/pixel thr "
                  f"{p['map_max_ratio']:.2f}); PatchCore {c['decision']} (score/thr "
                  f"{c['score_ratio']:.2f}, map max/pixel thr {c['map_max_ratio']:.2f}).", "",
                  caption, ""]
        print(name, info)
    if "all" in args.cases:
        (OUT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
