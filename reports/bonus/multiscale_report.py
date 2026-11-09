"""Build the multi-scale report from metrics.json only; never inspect labels/masks.

Run from the worktree root after multiscale_reproduce.sh completes.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

RUNS = ["patchcore_wrn50", "patchcore_ms_baseline_max", "patchcore_ms_l123",
        "patchcore_ms_l234", "patchcore_ms_res_1_1.5_mean", "patchcore_ms_res_1_1.5_zmean"]
CATEGORIES = ["bottle", "capsule", "screw", "carpet", "tile", "leather"]
METRICS = ["image_auroc", "image_f1", "pixel_auroc", "aupro", "fit_seconds",
           "infer_ms_per_image", "peak_gpu_mem_gb"]
REPORT = Path("reports/bonus/multiscale.md")


def main():
    records = {run: {category: json.loads((Path("results") / run / category / "metrics.json").read_text())
                     for category in CATEGORIES} for run in RUNS}
    with tempfile.TemporaryDirectory(prefix="multiscale-aggregate-") as temporary:
        root = Path(temporary)
        subprocess.run([sys.executable, "-m", "ad.aggregate", "--runs", *RUNS,
                        "--out", str(root / "tables")], check=True)
        for source, destination in [(root / "tables", Path("reports/bonus/multiscale_tables")),
                                    (root / "figures", Path("reports/bonus/multiscale_figures"))]:
            destination.mkdir(parents=True, exist_ok=True)
            for artifact in source.iterdir():
                shutil.copy2(artifact, destination / artifact.name)

    suite_summary = Path("reports/bonus/multiscale_pytest.txt").read_text().strip().splitlines()[-1]
    means = {}
    lines = [REPORT.read_text().split("Results, costs, analysis, and validation")[0].split("## Results")[0].rstrip() + "\n",
             "## Results\n",
             "Every mean is an unweighted category mean: objects = bottle/capsule/screw; "
             "textures = carpet/tile/leather. All 36 run/category combinations are present. "
             "AUROC, F1, and AU-PRO are fractions; fit time is seconds, inference is milliseconds "
             "per image, and peak memory is GiB of allocated CUDA tensors. Image F1 uses only "
             "normal-validation thresholds (99th percentile); pixel thresholds use the 99.9th "
             "percentile. The aggregate's `image_f1_max_oracle` is an oracle reference, not a "
             "threshold used by these runs.\n"]
    for run in RUNS:
        means[run] = {}
        lines.extend([f"### `{run}`\n", "| Category | Image AUROC | Image F1 | Pixel AUROC | AU-PRO | Fit s | Infer ms/image | Peak GiB |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        groups = [(category, [category]) for category in CATEGORIES] + [
            ("mean (objects)", CATEGORIES[:3]), ("mean (textures)", CATEGORIES[3:]),
            ("mean (all)", CATEGORIES)]
        for name, categories in groups:
            values = {metric: float(np.mean([records[run][c][metric] for c in categories])) for metric in METRICS}
            if name.startswith("mean"):
                means[run][name] = values
            formatted = [f"{values[metric]:.4f}" if index < 4 else f"{values[metric]:.3f}"
                         for index, metric in enumerate(METRICS)]
            lines.append("| " + " | ".join([name, *formatted]) + " |")
        lines.append("")

    lines.extend(["## Cost by scale\n", "Pyramid timings below include input resizing, feature extraction and PatchCore's "
                  "nearest-neighbour search (including its unused Eq. 7 score). They exclude map alignment/fusion "
                  "and shared smoothing, which remain included in the main inference column. Times are CUDA "
                  "synchronised and measured on preloaded test batches; loader/I/O latency is reported "
                  "separately by aggregate as `infer_ms_per_image_e2e`. Bank sizes below exclude the frozen "
                  "backbone and temporary tensors; the main table reports the complete run's peak allocation.\n",
                  "| Fusion | Category | Scale | Grid | Bank patches | Bank MiB | Fit s | Infer ms/image | Fit mean | Fit std |",
                  "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for run in RUNS[-2:]:
        for category in CATEGORIES:
            for scale in records[run][category]["model_stats"]["scales"]:
                grid = "x".join(map(str, scale["feature_grid"]))
                mean, std = scale["normalization"]
                reference = (f"{mean:.4f} | {std:.4f}" if run.endswith("zmean") else "— | —")
                lines.append(f'| {records[run][category]["model_stats"]["fusion"]} | {category} | '
                             f'{scale["scale"]:.1f} | {grid} | {scale["memory_bank_size"]} | '
                             f'{scale["memory_bank_bytes"] / 1024**2:.3f} | {scale["fit_seconds"]:.3f} | '
                             f'{scale["infer_ms_per_image"]:.3f} | {reference} |')
    lines.extend(["", "At fixed training-image count, layer1 gives four times as many patches/bank entries "
                  "as layer2; with a 10% greedy coreset the projected scan work grows roughly 16-fold "
                  "(candidate count times selection count). The 1.5 input scale has 2.25 times as many "
                  "patches, giving a combined 3.25 times baseline bank storage; its own greedy selection "
                  "work is roughly 5.06 times baseline. Calibrating zmean additionally searches each coreset "
                  "against the full training bank; the full bank is discarded after fit.\n"])

    def overall(run, metric):
        return means[run]["mean (all)"][metric]

    def delta(run, reference, metric):
        return overall(run, metric) - overall(reference, metric)

    baseline, maximum, l123, l234, mean, zmean = RUNS
    lines.append(f"Measured layer1 mean fit and inference costs were "
                 f"{overall(l123, 'fit_seconds') / overall(baseline, 'fit_seconds'):.2f}x and "
                 f"{overall(l123, 'infer_ms_per_image') / overall(baseline, 'infer_ms_per_image'):.2f}x "
                 "the baseline, respectively; feature extraction and fixed overheads make total costs "
                 "scale differently from the coreset scan bound; mean peak allocation rose "
                 f"{overall(l123, 'peak_gpu_mem_gb') / overall(baseline, 'peak_gpu_mem_gb'):.2f}x "
                 f"({overall(l123, 'peak_gpu_mem_gb'):.2f} versus "
                 f"{overall(baseline, 'peak_gpu_mem_gb'):.2f} GiB).\n")
    lines.append(f"Raw pyramid mean used "
                 f"{overall(mean, 'infer_ms_per_image') / overall(maximum, 'infer_ms_per_image'):.2f} "
                 f"times the max control's mean inference time and "
                 f"{overall(mean, 'peak_gpu_mem_gb') / overall(maximum, 'peak_gpu_mem_gb'):.2f} "
                 f"times its mean peak allocation; zmean fit took "
                 f"{overall(zmean, 'fit_seconds') / overall(mean, 'fit_seconds'):.2f} "
                 "times raw-mean fit because calibration performs an additional full-bank search.\n")
    def category_delta(run, reference, category, metric):
        return records[run][category][metric] - records[reference][category][metric]
    std_ratios = [record["model_stats"]["scales"][1]["normalization"][1]
                  / record["model_stats"]["scales"][0]["normalization"][1]
                  for record in records[zmean].values()]
    lines.extend(["## Analysis\n",
        f"Adding layer1 changed overall pixel AUROC by {delta(l123, baseline, 'pixel_auroc'):+.4f} "
        f"and AU-PRO by {delta(l123, baseline, 'aupro'):+.4f} relative to the Eq. 7 baseline, "
        "but these localisation gains come with image-level detection losses.",
        "The localisation-versus-detection trade-off is clearest on capsule: image AUROC fell "
        f"from {records[baseline]['capsule']['image_auroc']:.3f} to {records[l123]['capsule']['image_auroc']:.3f} "
        f"and F1 from {records[baseline]['capsule']['image_f1']:.3f} to {records[l123]['capsule']['image_f1']:.3f}; "
        f"tile F1 fell from {records[baseline]['tile']['image_f1']:.3f} to {records[l123]['tile']['image_f1']:.3f}, "
        f"and mean F1 from {overall(baseline, 'image_f1'):.3f} to {overall(l123, 'image_f1'):.3f}.",
        f"Layer1's AU-PRO gains were {category_delta(l123, baseline, 'screw', 'aupro'):+.4f} on screw, "
        f"{category_delta(l123, baseline, 'capsule', 'aupro'):+.4f} on capsule, "
        f"{category_delta(l123, baseline, 'carpet', 'aupro'):+.4f} on carpet and "
        f"{category_delta(l123, baseline, 'tile', 'aupro'):+.4f} on tile, so both of these textures "
        "also benefited in localisation.",
        f"Adding layer4 instead changed overall image AUROC by {delta(l234, baseline, 'image_auroc'):+.4f} "
        f"and AU-PRO by {delta(l234, baseline, 'aupro'):+.4f}; a possible explanation is that "
        "its coarse 7x7 layer4 features dilute fine defect evidence when the fused embedding "
        "is pooled back to 1024 dimensions.",
        f"Relative to the max-scoring control, raw pyramid mean changed image AUROC by "
        f"{delta(mean, maximum, 'image_auroc'):+.4f}, pixel AUROC by "
        f"{delta(mean, maximum, 'pixel_auroc'):+.4f}, and AU-PRO by {delta(mean, maximum, 'aupro'):+.4f}.\n",
        "Original-resolution median GT areas from the [dataset analysis on main]"
        "(../dataset/dataset_stats.md), compared with the fixed pyramid's category-level results\n",
        "| Category | Median defect area | Max-control AU-PRO | Pyramid mean AU-PRO | Change |",
        "| --- | ---: | ---: | ---: | ---: |"])
    area_medians = {"screw": 0.29, "capsule": 0.43, "carpet": 1.48, "tile": 7.93}
    for category, area in area_medians.items():
        lines.append(f"| {category} | {area:.2f}% | {records[maximum][category]['aupro']:.4f} | "
                     f"{records[mean][category]['aupro']:.4f} | "
                     f"{category_delta(mean, maximum, category, 'aupro'):+.4f} |")
    lines.extend(["",
        "These gains are not concentrated in the smallest-defect categories: carpet and tile "
        "improve at least as much as screw and capsule, although whole-category metrics do not "
        "isolate capsule scratches (median original defect area 0.64%) or control for other "
        "category differences.",
        "Within each category, zmean is a positive affine transform of `r*d1.0 + d1.5`, where "
        "the `d` terms are aligned raw distance maps and `r = std1.5 / std1.0`: subtracting "
        "the scale means adds only a uniform offset, so only "
        "the std ratio can change rankings and hence AUROC or AU-PRO.",
        "That ratio is about 1.03–1.09 in the reviewer's bottle/screw/tile diagnostic "
        f"({min(std_ratios):.2f}–{max(std_ratios):.2f} across all six categories), making zmean "
        "effectively equivalent to mean here: image AUROC and F1 are identical, and mean AU-PRO "
        f"changes by only {delta(zmean, mean, 'aupro'):+.4f}.",
        "The reviewer's held-out normal-validation diagnostic found biased absolute proxy "
        "statistics (tile mean 0.75 versus 1.24 held-out; screw std 0.18 versus 0.10), while "
        "the std ratio is roughly right with up to about 11% relative error, including tile's "
        "proxy ratio 1.09 versus held-out 0.98 in the opposite direction from unity.",
        "The mismatch arises because calibration queries k-center-selected coreset patches "
        "against the full training bank with same-image spatial neighbours retained, whereas "
        "prediction queries all patches against the coreset; the validation diagnostic was "
        "used only to explain the existing result and never for fitting, calibration or tuning.\n",
        "## Reproduction and validation\n",
        "Run `bash reports/bonus/multiscale_reproduce.sh` from the worktree root, then "
        "`PYTHONPATH=src /home/eva/code/ma6216/industrial-visual-anomaly-detection/.venv/bin/python "
        "reports/bonus/multiscale_report.py`. The reproduction script runs every category through "
        "the shared GPU lock and preserves baseline preprocessing; aggregate is generated in a "
        "temporary directory before copying to these task-specific report paths.\n",
        "- Full suite, with `HF_HUB_OFFLINE=1`, `MVTEC_ROOT` set and `scripts/gpu_run.sh`: "
        f"**{suite_summary}**.\n"
        "- Focused CPU multi-scale tests: **18 passed, 1 deselected in 0.62s**.\n"
        "- Twenty-training-image bottle GPU/data smoke: **1 passed in 2.40s**.\n"
        "- Tests cover map/grid alignment, exact 1.0-scale memory-bank equivalence to plain PatchCore "
        "for mean and zmean, identical-scale raw and standardised map equivalence, source-row exclusion, "
        "fit-only frozen z statistics, constant references, invalid scales, and first-layer hierarchy grids.\n",
        "Aggregate tables and the labelled oracle reference are in [multiscale_tables/summary.md]"
        "(multiscale_tables/summary.md); all numeric resource tables and per-category values are in "
        "[metrics_long.csv](multiscale_tables/metrics_long.csv).\n",
        "![Image AUROC](multiscale_figures/bar_image_auroc.png)\n"
        "![Pixel AUROC](multiscale_figures/bar_pixel_auroc.png)\n"
        "![AU-PRO](multiscale_figures/bar_aupro.png)\n"])
    # The max-score control changes no fitted features or pixel processing.
    for category in CATEGORIES:
        for metric in ("pixel_auroc", "aupro"):
            assert records[baseline][category][metric] == records[maximum][category][metric], (category, metric)
    REPORT.write_text("\n".join(lines))
    print(json.dumps({run: means[run]["mean (all)"] for run in RUNS}, indent=2))


if __name__ == "__main__":
    main()
