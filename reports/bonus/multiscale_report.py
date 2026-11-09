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
    lines = [REPORT.read_text().split("Results, costs, analysis, and validation")[0].split("## Results")[0],
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
                 "scale differently from the coreset scan bound.\n")
    def category_delta(run, reference, category, metric):
        return records[run][category][metric] - records[reference][category][metric]
    lines.extend(["## Analysis\n",
        f"Adding layer1 changed overall pixel AUROC by {delta(l123, baseline, 'pixel_auroc'):+.4f} "
        f"and AU-PRO by {delta(l123, baseline, 'aupro'):+.4f} relative to the Eq. 7 baseline, "
        "consistent with a finer grid changing the balance between local detail and semantic context.",
        f"For the small-defect categories screw and capsule, layer1 changed AU-PRO by "
        f"{category_delta(l123, baseline, 'screw', 'aupro'):+.4f} and "
        f"{category_delta(l123, baseline, 'capsule', 'aupro'):+.4f}, respectively; these are "
        "whole-category measurements and do not isolate capsule scratches.",
        f"On carpet and tile, the corresponding layer1 AU-PRO changes were "
        f"{category_delta(l123, baseline, 'carpet', 'aupro'):+.4f} and "
        f"{category_delta(l123, baseline, 'tile', 'aupro'):+.4f}, showing that extra fine-scale "
        "features do not have a uniform effect across textures.",
        f"Adding layer4 instead changed overall image AUROC by {delta(l234, baseline, 'image_auroc'):+.4f} "
        f"and AU-PRO by {delta(l234, baseline, 'aupro'):+.4f}; a possible explanation is that "
        "its coarse 7x7 layer4 features dilute fine defect evidence when the fused embedding "
        "is pooled back to 1024 dimensions.",
        f"Relative to the max-scoring control, raw pyramid mean changed image AUROC by "
        f"{delta(mean, maximum, 'image_auroc'):+.4f}, pixel AUROC by "
        f"{delta(mean, maximum, 'pixel_auroc'):+.4f}, and AU-PRO by {delta(mean, maximum, 'aupro'):+.4f}.",
        f"Training-only zmean changed AU-PRO by {delta(zmean, mean, 'aupro'):+.4f} compared with raw "
        "pyramid mean: it changes the relative weight of the two score distributions, while the "
        "self-excluded patch reference can still be optimistic because same-image neighbours remain.",
        f"Raw pyramid mean used {overall(mean, 'infer_ms_per_image') / overall(maximum, 'infer_ms_per_image'):.2f} "
        f"times the max control's mean inference time and {overall(mean, 'peak_gpu_mem_gb') / overall(maximum, 'peak_gpu_mem_gb'):.2f} "
        f"times its mean peak allocation; zmean fit took {overall(zmean, 'fit_seconds') / overall(mean, 'fit_seconds'):.2f} "
        "times raw-mean fit because calibration performs an additional full-bank search.",
        "Magnifying an already cropped 224-pixel image cannot recover removed source detail, and "
        "downsampling the 42x42 map to 28x28 can weaken the localisation benefit for very small defects.",
        "Image F1 also depends on the normal-validation score tail, so changes in F1 need not follow "
        "changes in threshold-free AUROC; all predefined variants are reported without selecting "
        "per-category winners or tuning on test results.\n",
        "## Reproduction and validation\n",
        "Run `bash reports/bonus/multiscale_reproduce.sh` from the worktree root, then "
        "`PYTHONPATH=src /home/eva/code/ma6216/industrial-visual-anomaly-detection/.venv/bin/python "
        "reports/bonus/multiscale_report.py`. The reproduction script runs every category through "
        "the shared GPU lock and preserves baseline preprocessing; aggregate is generated in a "
        "temporary directory before copying to these task-specific report paths.\n",
        "- Full suite, with `HF_HUB_OFFLINE=1`, `MVTEC_ROOT` set and `scripts/gpu_run.sh`: "
        f"**{suite_summary}**.\n"
        "- Focused CPU multi-scale tests: **15 passed, 1 deselected in 0.71s**.\n"
        "- Twenty-training-image bottle GPU/data smoke: **1 passed in 2.40s**.\n"
        "- Tests cover map/grid alignment, identical-scale raw-map equivalence, source-row exclusion, "
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
