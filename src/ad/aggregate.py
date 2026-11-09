"""Collect results/<run>/<category>/metrics.json into comparison tables and plots.

Owner: branch feat/pipeline.

CLI: python -m ad.aggregate [--results results] [--out reports/tables]
Writes metrics_long.csv, one Markdown table per metric (rows: category + mean,
object mean, texture mean; columns: runs), and grouped bar charts.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
