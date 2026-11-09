"""Collect results/<run>/<category>/metrics.json into comparison tables and plots.

Owner: branch feat/pipeline.

CLI: python -m ad.aggregate [--results results] [--out reports/tables] [--runs run1 run2]
Writes metrics_long.csv, one Markdown table per metric (rows: category + mean,
object mean, texture mean; columns: runs), and grouped bar charts.
Charts are written to <out>/../figures/bar_<metric>.png. Missing values are shown
as an em dash; mean cells include the number of categories with finite metric
values as [n=...]. A mean marked * lacks at least one of the six standard categories
with a finite metric value and must not be compared silently against a complete run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ad.data import CATEGORIES, TEXTURE_CATEGORIES

KEY_METRICS = (
    "image_auroc", "image_f1", "image_f1_max_oracle", "pixel_auroc", "aupro",
    "fit_seconds", "infer_ms_per_image", "infer_ms_per_image_e2e", "peak_gpu_mem_gb",
)
MAIN_METRICS = KEY_METRICS[:5]
MEAN_ROWS = ("mean (objects)", "mean (textures)", "mean (all)")


def _read_results(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.glob("*/*/metrics.json")):
        result = json.loads(path.read_text())
        category, run = path.parent.name, path.parent.parent.name
        for metric, value in result.items():
            if value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
                records.append({"run": run, "category": category,
                                "is_texture": bool(result.get("is_texture", category in TEXTURE_CATEGORIES)),
                                "metric": metric, "value": value})
    return pd.DataFrame(records, columns=["run", "category", "is_texture", "metric", "value"])


def _metric_table(frame: pd.DataFrame, metric: str, runs: list[str], categories: list[str]) -> pd.DataFrame:
    values = frame.loc[frame["metric"] == metric].copy()
    values["value"] = values["value"].replace([np.inf, -np.inf], np.nan)
    table = values.pivot(index="category", columns="run", values="value").reindex(
        index=categories, columns=runs)
    groups = (values.loc[~values["is_texture"].astype(bool)],
              values.loc[values["is_texture"].astype(bool)], values)
    finite = values.loc[values["value"].notna()]
    incomplete = [run for run in runs
                  if not set(CATEGORIES).issubset(set(finite.loc[finite["run"] == run, "category"]))]
    notes = {}
    for name, group in zip(MEAN_ROWS, groups):
        table.loc[name] = group.groupby("run")["value"].mean().reindex(runs)
        counts = group.groupby("run")["value"].count().reindex(runs, fill_value=0)
        notes[name] = {run: f"{'*' if run in incomplete else ''} [n={counts[run]}]" for run in runs}
    table.attrs["cell_notes"] = notes
    table.attrs["incomplete_runs"] = incomplete
    return table


def _markdown(table: pd.DataFrame, index_name: str = "category") -> str:
    # Avoid the optional tabulate dependency required by DataFrame.to_markdown.
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join([index_name, *(escape(column) for column in table.columns)]) + " |",
             "| " + " | ".join(["---", *(["---:"] * len(table.columns))]) + " |"]
    for name, row in table.iterrows():
        notes = table.attrs.get("cell_notes", {}).get(name, {})
        formatted = [(f"{value:.4f}" if pd.notna(value) and math.isfinite(value) else "—")
                     + notes.get(column, "") for column, value in row.items()]
        lines.append("| " + " | ".join([escape(name), *formatted]) + " |")
    if table.attrs.get("incomplete_runs"):
        lines.append("\n\\* Incomplete mean: at least one of the six standard categories lacks a finite result; "
                     "only available categories are averaged.")
    return "\n".join(lines) + "\n"


def _bar_chart(table: pd.DataFrame, metric: str, categories: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(categories)), 4.8))
    positions = np.arange(len(categories))
    width = 0.8 / len(table.columns)
    for index, run in enumerate(table.columns):
        ax.bar(positions + (index - (len(table.columns) - 1) / 2) * width,
               table.loc[categories, run].to_numpy(dtype=float), width=width, label=run)
    ax.set_xticks(positions, categories, rotation=20, ha="right")
    ax.set_ylabel({"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC", "aupro": "AU-PRO"}[metric])
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("reports/tables"))
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args(argv)
    frame = _read_results(args.results)
    available = sorted(frame["run"].unique())
    runs = list(dict.fromkeys(args.runs)) if args.runs else available
    if not runs:
        parser.error(f"no metrics.json results found under {args.results}")
    missing = set(runs) - set(available)
    if missing:
        parser.error(f"runs not found: {', '.join(sorted(missing))}")
    frame = frame.loc[frame["run"].isin(runs)].copy()
    # CSV and table columns follow the requested run ordering.
    frame["run"] = pd.Categorical(frame["run"], categories=runs, ordered=True)
    frame = frame.sort_values(["run", "category", "metric"])
    categories = [*CATEGORIES, *sorted(set(frame["category"]) - set(CATEGORIES))]
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "metrics_long.csv", index=False)
    tables = {}
    for metric in KEY_METRICS:
        table = _metric_table(frame, metric, runs, categories)
        tables[metric] = table
        note = "\nOracle reference only; uses test labels to select a threshold.\n" if metric == "image_f1_max_oracle" else ""
        (args.out / f"{metric}.md").write_text(_markdown(table) + note)
        if metric in ("image_auroc", "pixel_auroc", "aupro"):
            _bar_chart(table, metric, categories, figures / f"bar_{metric}.png")

    summary = ["# Experiment comparison\n",
               "Mean cells show category counts [n=...]; missing results are shown as —.\n",
               "image_f1 uses normal-validation thresholds; image_f1_max_oracle is an oracle reference.\n"]
    overall = pd.DataFrame({metric: tables[metric].loc["mean (all)"] for metric in KEY_METRICS})
    overall.attrs["cell_notes"] = {
        run: {metric: tables[metric].attrs["cell_notes"]["mean (all)"][run] for metric in KEY_METRICS}
        for run in runs}
    overall.attrs["incomplete_runs"] = sorted({run for table in tables.values()
                                              for run in table.attrs["incomplete_runs"]})
    summary.extend(["## Run means\n", _markdown(overall, "run")])
    for metric in MAIN_METRICS:
        summary.extend([f"## {metric}\n", _markdown(tables[metric])])
    (args.out / "summary.md").write_text("\n".join(summary))


if __name__ == "__main__":
    main()
