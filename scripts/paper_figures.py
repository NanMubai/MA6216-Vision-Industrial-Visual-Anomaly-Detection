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
from matplotlib.ticker import NullFormatter
from PIL import Image, ImageDraw, ImageFont

from paper_tables import (CATEGORIES, ROOT, RUNS, VARIANTS, csv_rows, load_metrics,
                          source_bytes, variant_value)

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
    for ax, metric, title, low in zip(axes, ("image_auroc", "aupro"),
                                    ("(a) Image AUROC", "(b) AU-PRO"), (0.78, 0.75)):
        for i, (method, run) in enumerate(RUNS.items()):
            vals = [data["baseline"].get(run, metric, (cat,)) for cat in CATEGORIES]
            ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=COLORS[i],
                   label=("PaDiM", "PatchCore")[i], hatch=("///", "")[i],
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.set(ylim=(low, 1.006), xticks=x, xticklabels=CATEGORIES, title=title)
        ax.tick_params(axis="x", rotation=25)
        ax.axvline(2.5, linestyle="--", color="0.35", linewidth=0.7)
        ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95,
                   facecolor="white", edgecolor="none", borderpad=0.3)
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
    ax.axvline(64 / 224**2 * 100, color=COLORS[1], linestyle="--", linewidth=1,
               label="1 stride-8 cell (0.128%)")
    ax.set(xscale="log", xlabel="GT defect area (% of original image)")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    save(fig, output / "fig_defect_area.pdf")


def region_size(output):
    rows = csv_rows("reports/figures/failure_cases/tables/region_detection_pooled.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.25), layout="constrained")
    x = np.arange(len(rows))
    for i, method in enumerate(("PaDiM", "PatchCore")):
        ax.plot(x, [float(r[method]) for r in rows], color=COLORS[i],
                marker=("o", "s")[i], markersize=4, linestyle=("--", "-")[i], label=method)
    for i, r in enumerate(rows):
        ax.text(i, 1.035, "$n=" + r["n"] + "$", ha="center", fontsize=7)
    ax.set(xticks=x, xticklabels=("<1", "1–3", "3–10", "10–30", "30–100", "≥100"),
           xlabel="GT region size (stride-8 cells)", ylabel="Region detection rate",
           ylim=(0.25, 1.13))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    save(fig, output / "fig_regionsize.pdf")


def tradeoff(output, data):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.95), layout="constrained")
    markers = {"baseline": "*", "dinov2": "o", "multiscale": "^", "postproc": "s"}
    colors = {"baseline": "black", "dinov2": COLORS[0], "multiscale": COLORS[1], "postproc": "0.45"}
    # Label coincident controls once; all 17 points are still plotted.
    labels = {
        "Base": ("WRN50", (-7, -27)), "DinoB": ("ViT-B", (7, -2)),
        "DinoS": ("ViT-S", (-20, -20)), "WrnDino": ("WRN+ViT-B", (5, -9)),
        "DinoNoLN": ("No-LN†", (5, -15)),
        "MsLOneTwoThree": ("L1+2+3", (-43, 9)),
        "MsLTwoThreeFour": ("L2+3+4", (5, 3)),
        "PyrMean": ("Pyr / z-Pyr", (-51, 13)),
        "PpTopk": ("Score controls", (7, -2)),
        "PpSigZero": ("σ=0,2", (7, 15)), "PpSigEight": ("σ=8", (9, -14)),
    }
    for v in VARIANTS:
        x, y = (variant_value(v, metric, data) for metric in ("Infer", "Aupro"))
        ax.scatter(x, y, marker=markers[v.family], s=37 if v.family != "baseline" else 70,
                   facecolors="white" if v.posthoc else colors[v.family],
                   edgecolors=colors[v.family], linewidths=0.8,
                   zorder=6 if v.family == "baseline" else 4)
        if v.macro in labels:
            label, offset = labels[v.macro]
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=7,
                        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.45}, zorder=5)
    for family, label in (("baseline", "Baseline"), ("dinov2", "DINOv2"),
                          ("multiscale", "Multiscale"), ("postproc", "Post-proc.")):
        ax.scatter([], [], color=colors[family], marker=markers[family], s=25, label=label)
    ax.set(xscale="log", xlim=(2.3, 35), ylim=(0.69, 0.978),
           xlabel="Inference (ms/image, log scale)", ylabel="AU-PRO")
    ax.set_xticks((3, 6, 12, 24), labels=("3", "6", "12", "24"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(color="0.9", linewidth=0.5)
    ax.legend(loc="center right", frameon=False, fontsize=7, handletextpad=0.3)
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
