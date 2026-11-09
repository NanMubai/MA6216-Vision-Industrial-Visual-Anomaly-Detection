"""Regenerate unchanged headline outputs, then append labelled exploratory results.

PYTHONPATH=src python reports/bonus/dinov2_aggregate.py
The only new run is post-hoc, test-informed (added after a screw diagnostic).
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from ad import aggregate
from ad.data import CATEGORIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEADLINE = ["patchcore_wrn50", "patchcore_dinov2_vitb14",
            "patchcore_dinov2_vits14", "patchcore_wrn50_dinov2"]
NEW_RUN = "patchcore_dinov2_vitb14_noln"
STATUS = "post-hoc, test-informed (added after a screw diagnostic)"
LABEL = f"{NEW_RUN} — {STATUS}"
PLOT_METRICS = ("image_auroc", "pixel_auroc", "aupro")


def sensitivity_plot(frame: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot all five systems, with the additional run explicitly exploratory."""
    runs = [*HEADLINE, LABEL]
    table = aggregate._metric_table(frame, metric, runs, list(CATEGORIES))
    names = ["WRN50 (a priori)", "DINOv2-B (a priori)",
             "DINOv2-S (a priori)", "WRN50+DINOv2-B (a priori)",
             f"DINOv2-B no final LN\n{STATUS}"]
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(CATEGORIES))
    width = 0.8 / len(runs)
    for index, (run, label) in enumerate(zip(runs, names)):
        ax.bar(positions + (index - 2) * width,
               table.loc[list(CATEGORIES), run].to_numpy(dtype=float),
               width=width, label=label)
    ax.set_xticks(positions, CATEGORIES)
    ax.set_ylabel({"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC",
                   "aupro": "AU-PRO"}[metric])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2,
              frameon=False, fontsize=9)
    fig.suptitle(f"Sensitivity comparison: {STATUS}", fontsize=11, y=0.98)
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.08, right=0.98)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("reports/bonus/dinov2_tables"))
    parser.add_argument("--figures", type=Path, default=Path("reports/bonus/dinov2_figures"))
    args = parser.parse_args()
    common = ["--results", str(args.results), "--out", str(args.out), "--runs"]
    # Use the same aggregate entry point and retain its original four-run tables.
    aggregate.main([*common, *HEADLINE])
    headline = {path.name: path.read_text() for path in args.out.glob("*.md")}
    staging = args.out.parent / "figures"
    headline_plots = {metric: (staging / f"bar_{metric}.png").read_bytes()
                      for metric in PLOT_METRICS}
    aggregate.main([*common, *HEADLINE, NEW_RUN])

    frame = aggregate._read_results(args.results)
    frame = frame.loc[frame["run"].isin([*HEADLINE, NEW_RUN])].copy()
    frame.loc[frame["run"] == NEW_RUN, "run"] = LABEL
    exploratory = frame.loc[frame["run"] == LABEL]
    tables = {key: aggregate._metric_table(exploratory, key, [LABEL], list(CATEGORIES))
              for key in aggregate.KEY_METRICS}
    section = f"\n## Post-hoc sensitivity (not used for selection)\n\n**{LABEL}**\n\n"
    for key, table in tables.items():
        oracle = ("\nOracle reference only; uses test labels to select a threshold.\n"
                  if key == "image_f1_max_oracle" else "")
        (args.out / f"{key}.md").write_text(
            headline[f"{key}.md"] + section + aggregate._markdown(table) + oracle)
    overall = pd.DataFrame({key: table.loc["mean (all)"] for key, table in tables.items()})
    overall.attrs["cell_notes"] = {
        LABEL: {key: table.attrs["cell_notes"]["mean (all)"][LABEL]
                for key, table in tables.items()}}
    summary = headline["summary.md"] + section + "### Run means\n\n"
    summary += aggregate._markdown(overall, "run")
    for key, table in tables.items():
        summary += f"\n### {key}\n\n" + aggregate._markdown(table)
        if key == "image_f1_max_oracle":
            summary += "\nOracle reference only; not a selection criterion.\n"
    (args.out / "summary.md").write_text(summary)

    # CSV also labels the exploratory run while retaining all numeric strings.
    csv_path = args.out / "metrics_long.csv"
    reader = csv.DictReader(io.StringIO(csv_path.read_text()))
    fields, rows = reader.fieldnames, list(reader)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row["run"] == NEW_RUN:
                row["run"] = LABEL
            writer.writerow(row)

    args.figures.mkdir(parents=True, exist_ok=True)
    for metric in PLOT_METRICS:
        (args.figures / f"bar_{metric}.png").write_bytes(headline_plots[metric])
        sensitivity_plot(frame, metric, args.figures / f"posthoc_bar_{metric}.png")
        (staging / f"bar_{metric}.png").unlink()
    if not any(staging.iterdir()):
        staging.rmdir()
    print(f"Preserved four-run headline tables/figures and appended {LABEL}.")


if __name__ == "__main__":
    main()
