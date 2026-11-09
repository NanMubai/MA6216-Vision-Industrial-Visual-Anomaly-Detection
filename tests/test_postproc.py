"""Postprocessing ablations and deployment thresholds, without GPU or test fitting."""
import json
import shutil
from datetime import datetime

import numpy as np
import pytest
import torch
import yaml

from ad import metrics, run, scoring, threshold
from ad.base import AnomalyModel, Prediction
from ad.config import load_config
from ad.data import CATEGORIES


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ControlledModel(AnomalyModel):
    def __init__(self, test_scale=1):
        super().__init__("cpu")
        self.test_scale = test_scale
        self.predictions = {}
        self.events = []

    def fit(self, loader):
        assert loader.dataset.split == "train"
        self.events.append("fit")

    def predict(self, loader):
        split = loader.dataset.split
        self.events.append(split)
        n = len(loader.dataset)
        maps = np.arange(n * 16, dtype=np.float32).reshape(n, 4, 4) / 16
        scores = np.arange(n, dtype=np.float32) * 4 + 10
        if split == "test":
            maps *= self.test_scale
            scores *= self.test_scale
        self.predictions[split] = Prediction(maps, scores)
        return self.predictions[split]

    def stats(self):
        return {"feature_grid": [4, 4]}


@pytest.fixture
def config_path(fake_mvtec, tmp_path):
    cfg = {"name": "pp_test", "method": "patchcore", "categories": ["toy"],
           "data": {"root": str(fake_mvtec), "resize": 40, "crop": 32,
                    "num_workers": 0, "batch_size": 2},
           "model": {"device": "cpu"}, "scoring": {"sigma": 2, "normalize": "zscore"}}
    path = tmp_path / "pp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_requested_variants_change_one_scoring_factor():
    base = load_config("configs/patchcore_wrn50.yaml")
    expected = {"topk1": "image_score", "max": "image_score", "sigma2": "sigma",
                "sigma8": "sigma", "nosmooth": "sigma", "zscore_global": "normalize"}
    for suffix, changed in expected.items():
        cfg = load_config(f"configs/patchcore_pp_{suffix}.yaml")
        assert {key for key in cfg["scoring"]
                if cfg["scoring"][key] != base["scoring"].get(key, "none")} == {changed}
        for key in base.keys() - {"name", "scoring"}:
            assert cfg[key] == base[key]


def test_sigma_and_one_percent_aggregation_take_effect():
    patches = np.zeros((1, 28, 28), dtype=np.float32)
    patches[0, 13, 13] = 100
    maps = [scoring.upsample_and_smooth(patches, 224, sigma, device="cpu")
            for sigma in (0, 2, 4, 8)]
    assert all(a.max() > b.max() for a, b in zip(maps, maps[1:]))
    max_score = scoring.image_scores_from_maps(maps[2], "max")[0]
    top_score = scoring.image_scores_from_maps(maps[2], "topk_mean", 0.01)[0]
    count = int(224 * 224 * 0.01)
    assert top_score == pytest.approx(np.sort(maps[2].ravel())[-count:].mean())
    assert top_score < max_score


@pytest.mark.parametrize("test_scale", [1, 10000])
def test_run_fits_zscore_before_test_and_only_on_val(config_path, tmp_path, monkeypatch, test_scale):
    model = ControlledModel(test_scale)
    monkeypatch.setattr(run, "build_model", lambda *args, **kwargs: model)
    original_fit = scoring.fit_zscore
    references = []

    def fit(ref):
        assert model.events == ["fit", "val"], "test must not be predicted before calibration"
        references.append(ref.copy())
        return original_fit(ref)

    monkeypatch.setattr(scoring, "fit_zscore", fit)
    run.main(["--config", str(config_path), "--out", str(tmp_path), "--no-figs"])
    directory = tmp_path / "pp_test" / "toy"
    result = json.loads((directory / "metrics.json").read_text())
    with np.load(directory / "maps.npz") as arrays:
        assert arrays["maps"].dtype == np.float32
        assert np.isfinite(arrays["maps"]).all()
        assert result["pixel_threshold_global"] == 3
        assert result["pixel_f1_global"] == metrics.f1_at_threshold(
            arrays["masks"], arrays["maps"], 3)["f1"]
    assert len(references) == 2
    np.testing.assert_array_equal(references[0], [10, 14])
    val_maps = scoring.upsample_and_smooth(model.predictions["val"].patch_maps, 32, 2, device="cpu")
    np.testing.assert_array_equal(references[1], val_maps)
    assert result["normalization"]["image"] == {"mean": 12, "std": 2, "scale": 2, "count": 2}
    assert result["normalization"]["pixel"] == original_fit(val_maps)
    with np.load(directory / "scores.npz") as arrays:
        np.testing.assert_array_equal(arrays["val_image_scores"], [-1, 1])
        expected = (model.predictions["test"].image_scores - 12) / 2
        np.testing.assert_array_equal(arrays["image_scores"], expected)
        assert result["image_f1_global"] == metrics.f1_at_threshold(arrays["labels"], expected, 3)["f1"]
    assert result["image_threshold_global"] == 3
    assert "image_f1_global_pooled" not in result
    assert result["global_pooled_status"] == "skipped_requires_standard_categories"
    summary = json.loads((tmp_path / "pp_test" / "summary.json").read_text())
    assert summary["mean"]["image_f1_global"] == result["image_f1_global"]
    assert not any("threshold" in key for key in summary["mean"])


def test_one_global_threshold_and_pooled_calibration_from_validation_only(tmp_path, monkeypatch):
    categories = {category: {} for category in CATEGORIES}
    vals = {category: np.array([-1, 0, 1]) if index == 0 else np.array([-2, 2])
            for index, category in enumerate(CATEGORIES)}
    scores = {category: np.array([0, 3, 4]) if index == 0 else np.array([3, 2, 0])
              for index, category in enumerate(CATEGORIES)}
    labels = np.array([0, 1, 1])
    for category in categories:
        directory = tmp_path / category
        directory.mkdir()
        np.savez(directory / "scores.npz", image_scores=scores[category], labels=labels,
                 val_image_scores=vals[category])
    original = threshold.global_image_threshold

    def checked(references, *args):
        for category, values in references.items():
            np.testing.assert_array_equal(values, vals[category])
        return original(references, *args)

    monkeypatch.setattr(threshold, "global_image_threshold", checked)
    run._apply_global_thresholds(tmp_path, categories)
    assert categories[CATEGORIES[0]]["image_f1_global"] == 1
    assert categories[CATEGORIES[1]]["image_f1_global"] == 0
    pooled = np.quantile(np.concatenate(list(vals.values())), 0.99)
    for category in categories:
        assert categories[category]["image_threshold_global"] == 3
        assert categories[category]["image_threshold_global_pooled"] == pytest.approx(pooled)
        assert categories[category]["image_f1_global_pooled"] == metrics.f1_at_threshold(
            labels, scores[category], pooled)["f1"]


def test_image_pixel_auroc_and_aupro_invariant_under_validation_zscore():
    rng = np.random.default_rng(9)
    labels = np.array([0, 1, 0, 1, 1, 0])
    masks = np.zeros((6, 12, 12), dtype=np.uint8)
    masks[[1, 3, 4], 3:8, 4:9] = 1
    maps = rng.normal(size=masks.shape).astype(np.float32) + 1.4 * masks
    image_scores = rng.normal(size=6).astype(np.float32) + labels
    # Calibration arrays have different size and distribution to every test array.
    val_scores = rng.normal(5, 2, size=21).astype(np.float32)
    val_maps = rng.normal(8, 3, size=(21, 12, 12)).astype(np.float32)
    zimages = scoring.zscore_normalize(image_scores, val_scores)
    zmaps = scoring.zscore_normalize(maps, val_maps)
    assert metrics.image_auroc(labels, image_scores) == metrics.image_auroc(labels, zimages)
    assert metrics.pixel_auroc(masks, maps) == metrics.pixel_auroc(masks, zmaps)
    assert metrics.aupro(masks, maps) == pytest.approx(metrics.aupro(masks, zmaps), abs=1e-7)


def test_prediction_cache_preserves_raw_outputs_and_rejects_other_models(config_path, tmp_path, monkeypatch):
    model = ControlledModel()
    monkeypatch.setattr(run, "build_model", lambda *args, **kwargs: model)
    cache = tmp_path / "cache"
    args = ["--config", str(config_path), "--out", str(tmp_path), "--no-figs",
            "--prediction-cache", str(cache)]
    run.main(args + ["--set", "scoring.normalize=none"])
    assert model.events == ["fit", "val", "test"]
    raw_result = json.loads((tmp_path / "pp_test" / "toy" / "metrics.json").read_text())
    metadata = json.loads((cache / "toy" / "metadata.json").read_text())
    assert metadata["source_run"] == "pp_test"
    assert datetime.fromisoformat(metadata["created_at"]).utcoffset().total_seconds() == 0
    version = metadata["identity"]["code_version"]
    assert version["torch"] == str(torch.__version__)
    assert version["timm"] == run.package_version("timm")
    assert {"backbones.py", "patchcore.py", "padim.py", "coreset.py", "data.py", "base.py"}.issubset(
        version["source_sha256"])
    assert all(len(value) == 64 for value in version["source_sha256"].values())
    assert "categories" not in metadata["identity"]
    assert raw_result["prediction_cache"]["fit_seconds_copied"] is False
    assert raw_result["prediction_cache"]["peak_gpu_mem_gb_copied"] is False
    with np.load(cache / "toy" / "val.npz") as arrays:
        np.testing.assert_array_equal(arrays["image_scores"], [10, 14])
        assert set(arrays.files) == {"patch_maps", "image_scores"}

    def unexpected_fit(*args, **kwargs):
        pytest.fail("cached postprocessing must not build/refit the model")

    monkeypatch.setattr(run, "build_model", unexpected_fit)
    run.main(args + ["--set", "name=normalized"])
    result = json.loads((tmp_path / "normalized" / "toy" / "metrics.json").read_text())
    assert result["prediction_cache"]["reused"] is True
    assert result["prediction_cache"]["source_run"] == "pp_test"
    assert result["prediction_cache"]["created_at"] == metadata["created_at"]
    assert result["prediction_cache"]["fit_seconds_copied"] is True
    assert result["prediction_cache"]["peak_gpu_mem_gb_copied"] is True
    assert result["fit_seconds"] == raw_result["fit_seconds"]
    assert result["peak_gpu_mem_gb"] == raw_result["peak_gpu_mem_gb"]
    # End-to-end: the same cached raw predictions go through both runner paths.
    for metric in ("image_auroc", "pixel_auroc", "aupro"):
        assert result[metric] == pytest.approx(raw_result[metric], abs=1e-7)
    assert result["normalization"]["image"]["mean"] == 12
    run.main(args + ["--set", "name=topk", "--set", "scoring.image_score=topk_mean"])
    with np.load(tmp_path / "topk" / "toy" / "scores.npz") as arrays:
        raw_maps = scoring.upsample_and_smooth(model.predictions["test"].patch_maps, 32, 2, device="cpu")
        raw_scores = scoring.image_scores_from_maps(raw_maps, "topk_mean", 0.01)
        val_maps = scoring.upsample_and_smooth(model.predictions["val"].patch_maps, 32, 2, device="cpu")
        val_scores = scoring.image_scores_from_maps(val_maps, "topk_mean", 0.01)
        np.testing.assert_array_equal(arrays["image_scores"], scoring.zscore_normalize(raw_scores, val_scores))
    with pytest.raises(SystemExit, match="1"):
        run.main(args + ["--set", "seed=999"])
    error = json.loads((tmp_path / "pp_test" / "toy" / "error.json").read_text())
    assert "cache model/data config differs" in error["error"]
    assert not (tmp_path / "pp_test" / "global_thresholds.json").exists()


@pytest.mark.parametrize("category_names", [CATEGORIES[:1], CATEGORIES[:-1], [*CATEGORIES, "custom"]])
def test_partial_or_custom_pool_warns_and_skips_pooled_metrics(tmp_path, monkeypatch, caplog, category_names):
    categories = {category: {"image_f1_global_pooled": 0.99, "image_threshold_global_pooled": 99}
                  for category in category_names}
    for category in categories:
        directory = tmp_path / category
        directory.mkdir()
        np.savez(directory / "scores.npz", image_scores=[0, 3, 4], labels=[0, 1, 1],
                 val_image_scores=[-1, 1])
    original = threshold.global_image_threshold

    def fixed_only(references, mode="z3"):
        assert mode == "z3", "a partial pool must never calibrate a pooled threshold"
        return original(references, mode)

    monkeypatch.setattr(threshold, "global_image_threshold", fixed_only)
    run._apply_global_thresholds(tmp_path, categories)
    assert "requires exactly the six standard categories" in caplog.text
    assert "pooled metrics are skipped" in caplog.text
    for category, result in categories.items():
        assert result["image_f1_global"] == 1
        assert result["global_threshold_categories"] == list(categories)
        assert not any(key.endswith("_global_pooled") for key in result)
        assert json.loads((tmp_path / category / "metrics.json").read_text()) == result
    metadata = json.loads((tmp_path / "global_thresholds.json").read_text())
    assert metadata["image_threshold_global_pooled"] is None
    assert metadata["pooled_threshold_categories"] == list(CATEGORIES)


def test_single_category_can_reuse_cache_from_multiple_categories(config_path, fake_mvtec, tmp_path, monkeypatch):
    shutil.copytree(fake_mvtec / "toy", fake_mvtec / "tile")
    monkeypatch.setattr(run, "build_model", lambda *args, **kwargs: ControlledModel())
    args = ["--config", str(config_path), "--out", str(tmp_path), "--no-figs",
            "--set", "scoring.normalize=none", "--prediction-cache", str(tmp_path / "cache")]
    run.main(args + ["--categories", "toy", "tile"])

    def unexpected_fit(*args, **kwargs):
        pytest.fail("changing the category list must not invalidate a per-category cache")

    monkeypatch.setattr(run, "build_model", unexpected_fit)
    run.main(args + ["--categories", "toy", "--set", "name=single"])
    result = json.loads((tmp_path / "single" / "toy" / "metrics.json").read_text())
    assert result["prediction_cache"]["reused"] is True
    assert result["prediction_cache"]["source_run"] == "pp_test"


@pytest.mark.parametrize("change", ["model", "source", "torch", "timm", "legacy"])
def test_cache_rejects_model_source_and_library_changes(config_path, tmp_path, monkeypatch, change):
    sources = tmp_path / "ad_sources"
    sources.mkdir()
    for path in run._CACHE_SOURCE_DIR.glob("*.py"):
        shutil.copy(path, sources / path.name)
    monkeypatch.setattr(run, "_CACHE_SOURCE_DIR", sources)
    monkeypatch.setattr(run, "build_model", lambda *args, **kwargs: ControlledModel())
    cache = tmp_path / "cache"
    args = ["--config", str(config_path), "--out", str(tmp_path), "--no-figs",
            "--prediction-cache", str(cache), "--set", "scoring.normalize=none"]
    run.main(args)
    overrides = []
    if change == "model":
        overrides = ["--set", "model.coreset_ratio=0.2"]
    elif change == "source":
        # Change a temporary source copy, never the actual shared model files.
        path = sources / "patchcore.py"
        path.write_text(path.read_text() + "\n# simulated model code change\n")
    elif change == "torch":
        monkeypatch.setattr(run.torch, "__version__", "changed")
    elif change == "timm":
        monkeypatch.setattr(run, "package_version", lambda name: "changed")
    else:
        path = cache / "toy" / "metadata.json"
        metadata = json.loads(path.read_text())
        del metadata["identity"]["code_version"]
        path.write_text(json.dumps(metadata))

    def unexpected_fit(*args, **kwargs):
        pytest.fail("stale cache must fail instead of silently rebuilding or replaying")

    monkeypatch.setattr(run, "build_model", unexpected_fit)
    with pytest.raises(SystemExit, match="1"):
        run.main(args + overrides)
    result_dir = tmp_path / "pp_test" / "toy"
    assert not (result_dir / "metrics.json").exists()
    error = json.loads((result_dir / "error.json").read_text())
    assert "cache model/data config differs or code/library version changed" in error["error"]


@pytest.mark.parametrize("reference", [[], [float("nan")], [float("inf")]])
def test_invalid_normalization_reference(reference):
    with pytest.raises(ValueError, match="ref"):
        scoring.fit_zscore(reference)


def test_global_threshold_validation():
    with pytest.raises(ValueError, match="mode"):
        threshold.global_image_threshold({}, "invalid")
    with pytest.raises(ValueError, match="required"):
        threshold.global_image_threshold({}, "pooled_quantile")
    with pytest.raises(ValueError, match="finite"):
        threshold.global_image_threshold({"empty": []}, "pooled_quantile")
    assert threshold.global_image_threshold({}) == 3
