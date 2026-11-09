import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ad import aggregate
from ad.data import CATEGORIES, TEXTURE_CATEGORIES


@pytest.fixture
def results_tree(tmp_path):
    root = tmp_path / "results"
    for run_index, name in enumerate(("padim", "patchcore")):
        for category_index, category in enumerate(CATEGORIES):
            directory = root / name / category
            directory.mkdir(parents=True)
            result = {metric: 0.5 + 0.1 * run_index + 0.01 * category_index for metric in aggregate.KEY_METRICS}
            result.update({"category": category, "is_texture": category in TEXTURE_CATEGORIES,
                           "config": {"name": name}, "model_stats": {"feature_dim": 10}})
            (directory / "metrics.json").write_text(json.dumps(result))
    # Run summaries and unrelated files must not be mistaken for category metrics.
    (root / "padim" / "summary.json").write_text('{"mean": {"image_auroc": 1}}')
    return root


def test_aggregate_tables_and_plots(results_tree, tmp_path):
    out = tmp_path / "reports" / "tables"
    aggregate.main(["--results", str(results_tree), "--out", str(out), "--runs", "patchcore", "padim"])
    long = pd.read_csv(out / "metrics_long.csv")
    assert list(long.columns) == ["run", "category", "is_texture", "metric", "value"]
    assert len(long) == 2 * 6 * len(aggregate.KEY_METRICS)
    assert long["run"].drop_duplicates().tolist() == ["patchcore", "padim"]
    assert set(long["category"]) == set(CATEGORIES)
    assert long.loc[long["category"].isin(TEXTURE_CATEGORIES), "is_texture"].all()
    for metric in aggregate.KEY_METRICS:
        text = (out / f"{metric}.md").read_text()
        assert "| category | patchcore | padim |" in text
        for category in (*CATEGORIES, *aggregate.MEAN_ROWS):
            assert f"| {category} |" in text
        assert "| mean (objects) | 0.6100 [n=3] | 0.5100 [n=3] |" in text
        assert "| mean (textures) | 0.6400 [n=3] | 0.5400 [n=3] |" in text
        assert "| mean (all) | 0.6250 [n=6] | 0.5250 [n=6] |" in text
        assert "Incomplete mean" not in text
    summary = (out / "summary.md").read_text()
    for metric in aggregate.MAIN_METRICS:
        assert metric in summary
    assert "oracle reference" in summary
    assert "Oracle reference only" in (out / "image_f1_max_oracle.md").read_text()
    for metric in ("image_auroc", "pixel_auroc", "aupro"):
        image = plt.imread(out.parent / "figures" / f"bar_{metric}.png")
        assert image.ndim == 3
        assert min(image.shape[:2]) > 100
    assert plt.get_fignums() == []


def test_partial_results_missing_metrics_and_run_selection(tmp_path):
    root = tmp_path / "results"
    for name in ("chosen", "excluded"):
        directory = root / name / "bottle"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({"image_auroc": 0.8, "aupro": None}))
    out = tmp_path / "tables"
    aggregate.main(["--results", str(root), "--out", str(out), "--runs", "chosen"])
    frame = pd.read_csv(out / "metrics_long.csv")
    assert frame["run"].unique().tolist() == ["chosen"]
    assert frame.loc[frame["metric"] == "aupro", "value"].isna().all()
    assert len(frame.loc[frame["metric"] == "aupro"]) == 1
    image = (out / "image_auroc.md").read_text()
    assert "| bottle | 0.8000 |" in image
    assert "| mean (objects) | 0.8000* [n=1] |" in image
    assert "| mean (textures) | —* [n=0] |" in image
    assert "| mean (all) | 0.8000* [n=1] |" in image
    assert "Incomplete mean" in image
    assert "| bottle | — |" in (out / "pixel_auroc.md").read_text()


def test_undefined_and_nonfinite_values_do_not_pollute_means(tmp_path):
    root = tmp_path / "results"
    for category, value in (("bottle", 0.8), ("capsule", np.nan), ("carpet", np.inf)):
        directory = root / "run" / category
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({"image_auroc": value}))
    out = tmp_path / "tables"
    aggregate.main(["--results", str(root), "--out", str(out)])
    table = (out / "image_auroc.md").read_text()
    assert "| mean (all) | 0.8000* [n=1] |" in table
    assert "| capsule | — |" in table


def test_partial_and_complete_means_show_counts_per_run(results_tree, tmp_path):
    (results_tree / "patchcore" / "leather" / "metrics.json").unlink()
    out = tmp_path / "tables"
    aggregate.main(["--results", str(results_tree), "--out", str(out), "--runs", "patchcore", "padim"])
    table = (out / "image_auroc.md").read_text()
    assert "| mean (all) | 0.6200* [n=5] | 0.5250 [n=6] |" in table
    assert "| mean (objects) | 0.6100* [n=3] | 0.5100 [n=3] |" in table
    assert "| mean (textures) | 0.6350* [n=2] | 0.5400 [n=3] |" in table
    assert "six standard categories" in table
    summary = (out / "summary.md").read_text()
    patchcore_row = next(line for line in summary.splitlines() if line.startswith("| patchcore |"))
    padim_row = next(line for line in summary.splitlines() if line.startswith("| padim |"))
    assert patchcore_row.count("* [n=5]") == len(aggregate.KEY_METRICS)
    assert padim_row.count(" [n=6]") == len(aggregate.KEY_METRICS)


def test_missing_results_and_unknown_runs(results_tree, tmp_path):
    with pytest.raises(SystemExit, match="2"):
        aggregate.main(["--results", str(tmp_path / "absent")])
    with pytest.raises(SystemExit, match="2"):
        aggregate.main(["--results", str(results_tree), "--runs", "absent"])
