#!/usr/bin/env python3
"""Create publication figures using only committed report CSVs and PNGs.

No model inference, dataset loading, metric computation or threshold fitting is
performed here. JPEG composites preserve the existing failure/qualitative panels.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from PIL import Image, ImageDraw, ImageFont

from paper_tables import (CATEGORIES, ROOT, RUNS, VARIANTS, csv_rows, load_metrics,
                          region_rates, source_bytes, variant_value)

COLORS = ("#0072B2", "#D55E00")  # Okabe--Ito blue and vermilion
SINGLE, DOUBLE = 3.5, 7.16
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.pad_inches": 0.02,
})


def save(fig, path):
    fig.savefig(path, metadata={"Creator": "scripts/paper_figures.py", "CreationDate": None})
    plt.close(fig)


def bars(output, data):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.05), layout="constrained")
    x = np.arange(len(CATEGORIES))
    for ax, metric, title in zip(axes, ("image_auroc", "aupro"),
                                 ("(a) Image AUROC", "(b) AU-PRO")):
        for i, (method, run) in enumerate(RUNS.items()):
            vals = [data["baseline"].get(run, metric, (cat,)) for cat in CATEGORIES]
            ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=COLORS[i],
                   label=("PaDiM", "PatchCore")[i], hatch=("///", "")[i],
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.set(ylim=(0.75, 1.006), xticks=x, xticklabels=CATEGORIES, title=title)
        ax.tick_params(axis="x", rotation=25)
        ax.axvline(2.5, linestyle="--", color="0.35", linewidth=0.7)
        ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    save(fig, output / "fig_bars.pdf")


def defect_area(output):
    rows = csv_rows("reports/dataset/defect_regions_per_image.csv")
    vals = [[float(r["area_frac"]) * 100 for r in rows if r["category"] == c]
            for c in CATEGORIES]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35), layout="constrained")
    boxes = ax.boxplot(vals, orientation="horizontal", tick_labels=CATEGORIES,
                       patch_artist=True, widths=0.55, showfliers=True,
                       flierprops={"marker": ".", "markersize": 2, "markeredgecolor": "0.4"},
                       medianprops={"color": "black", "linewidth": 1.1})
    for i, box in enumerate(boxes["boxes"]):
        box.set(facecolor="0.8" if i < 3 else "0.95", edgecolor="0.3")
    ax.invert_yaxis()
    # The source area_frac is measured on the full image, before centre crop.
    # One 8x8 cell of the 224 crop occupies 64/256^2 of the resized full image.
    ax.axvline(64 / 224**2 * (224 / 256)**2 * 100, color=COLORS[1], linestyle="--", linewidth=1,
               label="1 feature cell (≈0.098% of image)")
    ax.set(xscale="log", xlabel="GT defect area (% of full image)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=7)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    save(fig, output / "fig_defect_area.pdf")


def region_size(output):
    rows = csv_rows("reports/figures/failure_cases/tables/region_detection_pooled.csv")
    regions = csv_rows("reports/figures/failure_cases/tables/per_region.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.25), layout="constrained")
    x = np.arange(len(rows))
    for i, method in enumerate(("PaDiM", "PatchCore")):
        ax.plot(x, [float(region_rates(r, regions)[i]) for r in rows], color=COLORS[i],
                marker=("o", "s")[i], markersize=4, linestyle=("--", "-")[i], label=method)
    for i, r in enumerate(rows):
        ax.text(i, 1.035, "$n=" + r["n"] + "$", ha="center", fontsize=7)
    tick_labels = [r["size_bin"].removesuffix(" cells").removesuffix(" cell")
                   .replace(">=", "≥").replace("-", "–") for r in rows]
    ax.set(xticks=x, xticklabels=tick_labels, xlim=(-0.5, len(rows) - 0.5),
           xlabel="GT region size (stride-8 cells)", ylabel="Region detection rate",
           ylim=(0.25, 1.13))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    save(fig, output / "fig_regionsize.pdf")


def tradeoff(output, data):
    fig, ax = plt.subplots(figsize=(SINGLE, 3.15), layout="constrained")
    markers = {"baseline": "*", "dinov2": "o", "multiscale": "^", "postproc": "s"}
    colors = {"baseline": "black", "dinov2": COLORS[0], "multiscale": COLORS[1], "postproc": "0.45"}
    # Global thresholds affect F1, not AU-PRO; omit their coincident points.
    # Show the coarse-layer outlier with an explicit off-scale arrow so that
    # the useful AU-PRO range is legible at single-column width.
    labels = {
        "DinoB": ("ViT-B", (7, 2)),
        "DinoS": ("ViT-S", (-15, -12)), "WrnDino": ("WRN+ViT-B", (7, 0)),
        "DinoNoLN": ("No-LN†", (7, -8)),
        "MsLOneTwoThree": ("L1+2+3", (-35, 8)),
        "PyrMean": ("Pyr / z-Pyr", (-41, 11)),
        "PpSigEight": ("σ=8", (7, -8)),
    }
    for v in VARIANTS:
        if v.macro in ("PpGlobalZ", "PpGlobalPooled", "MsLTwoThreeFour"):
            continue
        x, y = (variant_value(v, metric, data) for metric in ("Infer", "Aupro"))
        ax.scatter(x, y, marker=markers[v.family], s=37 if v.family != "baseline" else 70,
                   facecolors="white" if v.posthoc else colors[v.family],
                   edgecolors=colors[v.family], linewidths=0.8,
                   zorder=6 if v.family == "baseline" else 4)
        if v.macro in labels:
            label, offset = labels[v.macro]
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=7,
                        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.45}, zorder=5)
    base = VARIANTS[0]
    ax.annotate("Baseline + post-proc.\n(σ≤4, max, top-1%)",
                (variant_value(base, "Infer", data), variant_value(base, "Aupro", data)),
                xytext=(2.6, 0.958), textcoords="data", ha="left", va="top", fontsize=7,
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.45})
    coarse = next(v for v in VARIANTS if v.macro == "MsLTwoThreeFour")
    ax.annotate(f"L2+3+4: {variant_value(coarse, 'Aupro', data):.3f}\n(below axis)",
                (variant_value(coarse, "Infer", data), 0), xycoords=ax.get_xaxis_transform(),
                xytext=(0, 19), textcoords="offset points", ha="left", va="bottom", fontsize=7,
                arrowprops={"arrowstyle": "->", "color": colors["multiscale"], "lw": 0.8})
    handles = [Line2D([], [], color=colors[family], marker=markers[family], linestyle="none", markersize=5, label=label)
               for family, label in (("baseline", "Baseline"), ("dinov2", "DINOv2"),
                                     ("multiscale", "Multiscale"), ("postproc", "Post-proc."))]
    handles.append(Line2D([], [], marker="o", color=COLORS[0], markerfacecolor="white",
                          linestyle="none", markersize=5, label="Hollow: post-hoc"))
    ax.set(xscale="log", xlim=(2.3, 35), ylim=(0.895, 0.963),
           xlabel="Inference (ms/image, log scale)", ylabel="AU-PRO")
    ax.set_xticks((3, 6, 12, 24), labels=("3", "6", "12", "24"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(color="0.9", linewidth=0.5)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=7, handletextpad=0.3, columnspacing=0.8)
    save(fig, output / "fig_tradeoff.pdf")


def panel(path):
    return Image.open(io.BytesIO(source_bytes(path))).convert("RGB")


def font(size):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", weight="bold"))
    return ImageFont.truetype(path, size)


def failure_composite(output, numbers, name):
    # Fixed dimensions retain every source panel's aspect ratio and full content.
    names = subprocess_report_pngs("reports/figures/failure_cases/")
    images = [panel(next(p for p in names if Path(p).name.startswith(f"{n:02d}_"))) for n in numbers]
    tile_w, tile_h, heading, gap = 1030, 375, 32, 12
    assert len(images) == 6, "failure composites require six panels"
    assert all(im.size == (tile_w, tile_h) for im in images), [im.size for im in images]
    canvas = Image.new("RGB", (2 * tile_w + gap, 3 * (tile_h + heading) + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(images):
        x, y = (i % 2) * (tile_w + gap), (i // 2) * (tile_h + heading + gap)
        draw.text((x + 5, y), f"({chr(97+i)})", fill="black", font=font(27))
        canvas.paste(im, (x, y + heading))
    canvas.save(output / name, quality=85, optimize=True, dpi=(290, 290))
    assert (output / name).stat().st_size <= 1_500_000


def subprocess_report_pngs(prefix):
    import subprocess
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD", "--", prefix], cwd=ROOT)
    return [p for p in names.decode().splitlines() if p.endswith(".png")]


def qualitative(output):
    for cat in CATEGORIES:
        ims = [panel(f"reports/figures/qualitative/{run}_{cat}_summary.png") for run in RUNS.values()]
        # 2,000 px / 7.16 in = 279 dpi; retain fine labels in the source grids.
        width, heading, gap = 994, 42, 12
        ims = [im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS) for im in ims]
        canvas = Image.new("RGB", (2 * width + gap, max(im.height for im in ims) + heading), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (label, im) in enumerate(zip(("PaDiM", "PatchCore"), ims)):
            x = i * (width + gap)
            draw.text((x + width / 2, 2), label, fill="black", font=font(30), anchor="mt")
            canvas.paste(im, (x, heading))
        canvas.save(output / f"fig_app_qual_{cat}.jpg", quality=85, optimize=True, dpi=(280, 280))


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    bars(output, data)
    defect_area(output)
    region_size(output)
    tradeoff(output, data)
    failure_composite(output, (1, 4, 5, 7, 8, 12), "fig_failures.jpg")
    failure_composite(output, range(1, 7), "fig_app_failures_1.jpg")
    failure_composite(output, range(7, 13), "fig_app_failures_2.jpg")
    qualitative(output)
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    assert total < 12_000_000, total
    print(f"{output}: {total / 1_000_000:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated/figures")
    generate(parser.parse_args().output_dir)
