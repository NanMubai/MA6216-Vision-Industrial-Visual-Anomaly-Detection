import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import weakref

import numpy as np
import pytest
import torch
import yaml

from ad import run, scoring
from ad.base import AnomalyModel, Prediction
from ad.data import MVTecDataset, get_root


class FakeModel(AnomalyModel):
    """Controlled predictions for the fixture; never consumes labels or masks."""

    def __init__(self, *, provided_scores=True, events=None, **kwargs):
        super().__init__(device="cpu")
        self.provided_scores = provided_scores
        self.events = events if events is not None else []
        self.predictions = {}
        self.fit_paths = []

    def fit(self, loader):
        assert loader.dataset.split == "train"
        self.events.append("fit")
        for batch in loader:
            self.fit_paths.extend(batch["path"])
            assert batch["image"].ndim == 4
        assert all("/train/good/" in path for path in self.fit_paths)

    def predict(self, loader):
        self.events.append(loader.dataset.split)
        maps = []
        for batch in loader:
            for path in batch["path"]:
                patch_map = np.random.random((8, 8)).astype(np.float32)
                # The fixture's known central crack corresponds to these cells.
                if "/test/crack/" in path:
                    patch_map[2:6, 3:5] += 10
                maps.append(patch_map)
        maps = np.stack(maps)
        scores = maps.max(axis=(1, 2)) + 5 if self.provided_scores else None
        prediction = Prediction(maps, scores)
        self.predictions[loader.dataset.split] = prediction
        return prediction

    def stats(self):
        return {"fit_image_count": len(self.fit_paths), "feature_grid": (8, 8)}


@pytest.fixture
def tiny_config(tmp_path, fake_mvtec, monkeypatch):
    monkeypatch.setenv("MVTEC_ROOT", str(fake_mvtec))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = {
        "name": "tiny", "method": "patchcore", "categories": ["toy"], "seed": 17,
        "data": {"resize": 40, "crop": 32, "batch_size": 2, "num_workers": 0},
        "model": {"device": "cpu"}, "scoring": {"sigma": 0.8},
        "threshold": {"image_quantile": 0.75, "pixel_quantile": 0.9},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    yield path
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("mode,provided_scores", [("model", True), ("model", False), ("topk_mean", True)])
def test_run_end_to_end(tiny_config, tmp_path, fake_mvtec, monkeypatch, mode, provided_scores):
    events, models = [], []

    def build(method, **kwargs):
        assert method == "patchcore"
        assert kwargs["seed"] == 17
        model = FakeModel(provided_scores=provided_scores, events=events)
        models.append(model)
        return model

    monkeypatch.setattr(run, "build_model", build)
    original_image = run.threshold.image_threshold
    original_pixel = run.threshold.pixel_threshold
    references = {}

    def image_threshold(values, quantile):
        events.append("image_threshold")
        references["scores"] = values.copy()
        return original_image(values, quantile)

    def pixel_threshold(values, quantile):
        events.append("pixel_threshold")
        references["maps"] = values.copy()
        return original_pixel(values, quantile)

    monkeypatch.setattr(run.threshold, "image_threshold", image_threshold)
    monkeypatch.setattr(run.threshold, "pixel_threshold", pixel_threshold)
    monkeypatch.setattr(run.visualize, "save_category_figures", lambda *a: pytest.fail("--no-figs ignored"))
    output = tmp_path / "results"
    run.main(["--config", str(tiny_config), "--out", str(output), "--max-train", "4", "--no-figs",
              "--set", f"scoring.image_score={mode}"])
    directory = output / "tiny" / "toy"
    result = json.loads((directory / "metrics.json").read_text())
    cfg = result["config"]
    model = models[0]
    assert len(model.fit_paths) == 4
    assert events == ["fit", "val", "image_threshold", "pixel_threshold", "test"]
    assert result["model_stats"] == {"fit_image_count": 4, "feature_grid": [8, 8]}
    assert result["category"] == "toy"
    assert result["is_texture"] is False
    assert cfg["data"]["root"] == str(fake_mvtec)
    assert cfg["data"]["max_train"] == 4

    with np.load(directory / "scores.npz", allow_pickle=False) as scores:
        assert set(scores.files) == {"image_scores", "labels", "defect_types", "paths", "val_image_scores"}
        for key in ("image_scores", "labels", "defect_types", "paths"):
            assert scores[key].shape == (5,)
        assert scores["val_image_scores"].shape == (2,)
        expected = MVTecDataset("toy", "test", root=fake_mvtec, resize=40, crop=32)
        assert scores["paths"].tolist() == [str(sample[0]) for sample in expected.samples]
        np.testing.assert_array_equal(scores["labels"], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(scores["defect_types"], ["crack", "crack", "good", "good", "good"])
        np.testing.assert_array_equal(scores["val_image_scores"], references["scores"])
        assert result["image_threshold"] == pytest.approx(np.quantile(scores["val_image_scores"], 0.75))
        val_paths = {str(sample[0]) for sample in MVTecDataset("toy", "val", root=fake_mvtec).samples}
        assert not val_paths.intersection(model.fit_paths)
        for split, saved_scores in (("val", scores["val_image_scores"]), ("test", scores["image_scores"])):
            prediction = model.predictions[split]
            maps = scoring.upsample_and_smooth(prediction.patch_maps, 32, 0.8, device="cpu")
            expected_scores = (prediction.image_scores if mode == "model" and provided_scores else
                               scoring.image_scores_from_maps(maps, "max" if mode == "model" else mode))
            np.testing.assert_allclose(saved_scores, expected_scores)

    with np.load(directory / "maps.npz", allow_pickle=False) as maps:
        assert set(maps.files) == {"maps", "masks"}
        assert maps["maps"].shape == maps["masks"].shape == (5, 32, 32)
        assert maps["maps"].dtype == np.float16
        assert maps["masks"].dtype == np.uint8
        np.testing.assert_array_equal(maps["masks"], np.stack([expected[i]["mask"][0].numpy() for i in range(5)]))
    expected_val = scoring.upsample_and_smooth(model.predictions["val"].patch_maps, 32, 0.8, device="cpu")
    np.testing.assert_allclose(references["maps"], expected_val)
    assert result["pixel_threshold"] == pytest.approx(np.quantile(expected_val, 0.9))
    for key in ("image_auroc", "image_f1", "image_f1_max_oracle", "pixel_auroc", "aupro", "pixel_f1", "pixel_iou"):
        assert 0 <= result[key] <= 1
    for key in ("fit_seconds", "infer_ms_per_image", "infer_ms_per_image_e2e", "peak_gpu_mem_gb"):
        assert result[key] >= 0
    assert result["infer_ms_per_image_e2e"] >= result["infer_ms_per_image"]
    assert result["peak_gpu_mem_gb"] == 0
    summary = json.loads((output / "tiny" / "summary.json").read_text())
    assert summary["categories"]["toy"] == result
    assert summary["mean"]["image_auroc"] == result["image_auroc"]
    assert summary["errors"] == {}
    assert "image_threshold" not in summary["mean"]
    assert "pixel_threshold" not in summary["mean"]


def test_latency_excludes_loading_and_metadata_avoids_second_image_pass(tiny_config, tmp_path, monkeypatch):
    clock = [0.0]
    test_reads = []
    original_getitem = MVTecDataset.__getitem__
    original_loader = run.make_loader
    original_smooth = scoring.upsample_and_smooth

    def getitem(dataset, index):
        batch = original_getitem(dataset, index)
        if dataset.split == "test":
            test_reads.append(index)
            # Saved annotations must come from samples, not batch metadata.
            batch.update(label=99, defect_type="wrong", path=batch["path"] + "?batch")
        return batch

    def loader(dataset, **kwargs):
        real_loader = original_loader(dataset, **kwargs)

        class DelayedLoader:
            def __init__(self):
                self.dataset = dataset

            def __iter__(self):
                if dataset.split == "test":
                    clock[0] += 50.0
                yield from real_loader

        return DelayedLoader()

    class TimedModel(FakeModel):
        def predict(self, loader):
            if loader.dataset.split == "test":
                clock[0] += 5.0
            return super().predict(loader)

    def smooth(patches, *args, **kwargs):
        if len(patches) == 5:
            clock[0] += 2.0
        return original_smooth(patches, *args, **kwargs)

    monkeypatch.setattr(MVTecDataset, "__getitem__", getitem)
    monkeypatch.setattr(run, "make_loader", loader)
    monkeypatch.setattr(run, "build_model", lambda *a, **kw: TimedModel())
    monkeypatch.setattr(scoring, "upsample_and_smooth", smooth)
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    run.main(["--config", str(tiny_config), "--out", str(tmp_path), "--no-figs"])
    directory = tmp_path / "tiny" / "toy"
    result = json.loads((directory / "metrics.json").read_text())
    assert result["infer_ms_per_image"] == pytest.approx(1000 * 7 / 5)
    assert result["infer_ms_per_image_e2e"] == pytest.approx(1000 * 57 / 5)
    assert test_reads == list(range(5))
    with np.load(directory / "scores.npz") as scores:
        np.testing.assert_array_equal(scores["labels"], [1, 1, 0, 0, 0])
        assert scores["defect_types"].tolist() == ["crack", "crack", "good", "good", "good"]
        assert all(not path.endswith("?batch") for path in scores["paths"])


@pytest.mark.parametrize("val_has_scores", [False, True])
@pytest.mark.parametrize("mode", ["model", "max"])
def test_inconsistent_image_score_source_is_a_category_error(
        tiny_config, tmp_path, monkeypatch, val_has_scores, mode):
    class InconsistentModel(FakeModel):
        def predict(self, loader):
            self.provided_scores = val_has_scores if loader.dataset.split == "val" else not val_has_scores
            return super().predict(loader)

    monkeypatch.setattr(run, "build_model", lambda *a, **kw: InconsistentModel())
    with pytest.raises(SystemExit, match="1"):
        run.main(["--config", str(tiny_config), "--out", str(tmp_path), "--no-figs",
                  "--set", f"scoring.image_score={mode}"])
    error = json.loads((tmp_path / "tiny" / "toy" / "error.json").read_text())
    assert error["error_type"] == "ValueError"
    assert "image_scores availability differs" in error["error"]
    assert "Traceback" in error["traceback"]
    summary = json.loads((tmp_path / "tiny" / "summary.json").read_text())
    assert summary["categories"] == summary["mean"] == {}
    assert summary["errors"]["toy"] == error


def test_failed_category_continues_writes_summary_and_clears_stale_metrics(
        tiny_config, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(run, "build_model", lambda *a, **kw: FakeModel())
    stale = tmp_path / "tiny" / "missing"
    stale.mkdir(parents=True)
    (stale / "metrics.json").write_text('{"image_auroc": 1}')
    with pytest.raises(SystemExit, match="1"):
        run.main(["--config", str(tiny_config), "--out", str(tmp_path), "--no-figs",
                  "--categories", "missing", "toy"])
    assert not (stale / "metrics.json").exists()
    error = json.loads((stale / "error.json").read_text())
    assert error["error_type"] == "FileNotFoundError"
    assert error["category"] == "missing"
    assert "Traceback" in caplog.text
    summary = json.loads((tmp_path / "tiny" / "summary.json").read_text())
    assert list(summary["categories"]) == ["toy"]
    assert summary["errors"] == {"missing": error}
    assert summary["mean"]["image_auroc"] == summary["categories"]["toy"]["image_auroc"]
    assert "image_threshold" not in summary["mean"]
    # A successful retry replaces an earlier error marker.
    successful = tmp_path / "tiny" / "toy"
    (successful / "error.json").write_text('{"error": "old failure"}')
    run.main(["--config", str(tiny_config), "--out", str(tmp_path), "--no-figs"])
    assert not (successful / "error.json").exists()


def test_set_overrides_and_visualize_stub(tiny_config, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(run, "build_model", lambda *a, **kw: FakeModel())

    def stub(*args):
        raise NotImplementedError

    monkeypatch.setattr(run.visualize, "save_category_figures", stub)
    run.main(["--config", str(tiny_config), "--out", str(tmp_path), "--set", "scoring.sigma=0",
              "--set", "name=overridden", "--set", "data.max_train=3", "--set", "categories=[toy]",
              "--set", "model.seed=23", "--set", "data.num_workers=0", "--categories", "toy"])
    cfg = json.loads((tmp_path / "overridden" / "toy" / "metrics.json").read_text())["config"]
    assert cfg["scoring"]["sigma"] == 0
    assert cfg["scoring"]["topk_ratio"] == 0.01  # untouched nested defaults survive
    assert cfg["data"]["max_train"] == 3
    assert cfg["model"]["seed"] == 23
    assert cfg["categories"] == ["toy"]
    assert "Visualization is not implemented" in caplog.text
    assert run._parse_overrides(["data.max_train=null", "flag=true", 'text="a=b"']) == {
        "data": {"max_train": None}, "flag": True, "text": "a=b"}


def test_multiple_categories_mean_and_model_cleanup(tiny_config, fake_mvtec, tmp_path, monkeypatch):
    shutil.copytree(fake_mvtec / "toy", fake_mvtec / "tile")
    models = []

    class ReversedScores(FakeModel):
        def predict(self, loader):
            prediction = super().predict(loader)
            if loader.dataset.split == "test":
                prediction.image_scores *= -1
            return prediction

    def build(*args, **kwargs):
        model = FakeModel() if not models else ReversedScores()
        models.append(weakref.ref(model))
        return model

    monkeypatch.setattr(run, "build_model", build)
    run.main(["--config", str(tiny_config), "--out", str(tmp_path), "--categories", "toy", "tile", "--no-figs"])
    summary = json.loads((tmp_path / "tiny" / "summary.json").read_text())
    assert list(summary["categories"]) == ["toy", "tile"]
    assert summary["categories"]["tile"]["is_texture"] is True
    assert summary["mean"]["image_auroc"] == pytest.approx(0.5)
    assert all(reference() is None for reference in models)


@pytest.mark.parametrize("assignment", ["missing_equals", ".sigma=1", "scoring..sigma=1", "x=[bad"])
def test_invalid_set_override(tiny_config, assignment):
    with pytest.raises(SystemExit, match="2"):
        run.main(["--config", str(tiny_config), "--set", assignment])


def test_invalid_category_override(tiny_config):
    with pytest.raises(SystemExit, match="2"):
        run.main(["--config", str(tiny_config), "--set", "categories=toy"])


@pytest.mark.gpu
@pytest.mark.data
def test_patchcore_gpu_data_smoke(tmp_path):
    if not (get_root() / "bottle" / "train" / "good").is_dir():
        pytest.skip("MVTec bottle data is missing (set MVTEC_ROOT)")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": "src"}
    # The test also works when the outer pytest process already owns the flock:
    # run directly here; callers must wrap GPU-marked pytest with gpu_run.sh.
    completed = subprocess.run(
        [sys.executable, "-m", "ad.run", "--config", "configs/patchcore_wrn50.yaml",
         "--categories", "bottle", "--max-train", "20", "--no-figs", "--out", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / "patchcore_wrn50" / "bottle" / "metrics.json").read_text())
    assert 0 <= result["image_auroc"] <= 1
