import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ad.base import build_model
from ad.config import load_config
from ad.data import MVTecDataset
from ad.multiscale import MultiResPatchCore
from ad.patchcore import PatchCore


class ImageOnlyBatch(dict):
    def __getitem__(self, key):
        assert key == "image", f"model accessed forbidden batch field {key!r}"
        return super().__getitem__(key)


class FakeBackbone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, images):
        strides = {"layer1": 1, "layer2": 2, "layer3": 4, "layer4": 8}
        return [F.avg_pool2d(images[:, :2], strides[layer]) for layer in self.layers]


@pytest.fixture
def fake_backbone(monkeypatch):
    monkeypatch.setattr("ad.patchcore.build_backbone", lambda name, layers: FakeBackbone(layers))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batches(images):
    return [ImageOnlyBatch(image=batch) for batch in images.split(2)]


def options():
    return dict(device="cpu", target_dim=4, proj_dim=2, patch_size=1,
                coreset_ratio=0.25, seed=5)


@pytest.mark.parametrize("fusion", ["mean", "zmean"])
def test_fusion_shapes_alignment_and_stats(fake_backbone, fusion):
    generator = torch.Generator().manual_seed(12)
    train = batches(torch.randn(3, 3, 8, 12, generator=generator))
    query = batches(torch.randn(2, 3, 8, 12, generator=generator))
    # The reference scale need not be the first configured scale.
    model = build_model("patchcore_multires", scales=[1.5, 1.0], fusion=fusion, **options())
    model.fit(train)
    frozen_stats = copy.deepcopy(model.normalization)
    aligned = []
    for scale, single, (mean, std) in zip(model.scales, model.models, model.normalization):
        images = query[0]["image"]
        resized = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
        raw = single.predict([ImageOnlyBatch(image=resized)]).patch_maps
        maps = torch.from_numpy((raw - mean) / std)
        aligned.append(F.interpolate(maps[:, None], size=(4, 6), mode="bilinear",
                                     align_corners=False)[:, 0])
    prediction = model.predict(query)
    assert [single.feature_grid for single in model.models] == [(6, 9), (4, 6)]
    assert prediction.patch_maps.shape == (2, 4, 6)
    assert prediction.patch_maps.dtype == np.float32
    assert prediction.image_scores is None
    np.testing.assert_allclose(prediction.patch_maps, torch.stack(aligned).mean(0), atol=1e-6)
    assert model.normalization == frozen_stats
    model.predict(batches(torch.full((1, 3, 8, 12), 1000.)))
    assert model.normalization == frozen_stats
    stats = model.stats()
    assert len(stats["scales"]) == 2
    assert stats["scales"][1]["memory_bank_size"] == 18
    assert stats["scales"][1]["memory_bank_bytes"] == 18 * 4 * 4
    assert all(item["fit_seconds"] >= 0 for item in stats["scales"])


def test_identical_scales_equal_single_scale_raw_maps(fake_backbone):
    generator = torch.Generator().manual_seed(4)
    train = batches(torch.randn(3, 3, 8, 8, generator=generator))
    query = batches(torch.randn(2, 3, 8, 8, generator=generator))
    single = PatchCore(**options())
    multiple = MultiResPatchCore(scales=[1.0, 1.0], **options())
    single.fit(train)
    multiple.fit(train)
    expected = single.predict(query)
    actual = multiple.predict(query)
    np.testing.assert_array_equal(actual.patch_maps, expected.patch_maps)
    assert actual.image_scores is None  # Eq. 7 is intentionally delegated to shared max scoring.


def test_zmean_reference_excludes_source_rows_and_uses_only_fit(fake_backbone):
    model = MultiResPatchCore(device="cpu", scales=[1.0, 1.0], fusion="zmean",
                             layers=["layer1"], target_dim=1, patch_size=1, coreset_ratio=1)
    train = torch.tensor([0., 2., 5.]).view(1, 1, 1, 3).expand(1, 3, 1, 3)
    model.fit(batches(train))
    # Full coreset: source-excluded distances are [2, 2, 3], not [0, 0, 0].
    expected = np.asarray([2., 2., 3.])
    for mean, std in model.normalization:
        assert mean == pytest.approx(expected.mean())
        assert std == pytest.approx(expected.std())
    before = copy.deepcopy(model.normalization)
    query = torch.full_like(train, 100.)
    result = model.predict(batches(query))
    mean, std = before[0]
    np.testing.assert_allclose(result.patch_maps, (95. - mean) / std, rtol=1e-6)
    assert model.normalization == before
    assert all(m.reference_count == 3 for m in model.models)


def test_constant_reference_is_finite_and_empty_prediction(fake_backbone):
    model = MultiResPatchCore(scales=[1.0], fusion="zmean", **options())
    model.fit(batches(torch.zeros(2, 3, 8, 8)))
    assert model.normalization == ((0., 1.),)
    assert np.isfinite(model.predict(batches(torch.ones(1, 3, 8, 8))).patch_maps).all()
    assert model.predict([]).patch_maps.shape == (0, 4, 4)


@pytest.mark.parametrize("layers,grid", [(["layer1", "layer2", "layer3"], (8, 8)),
                                          (["layer2", "layer3", "layer4"], (4, 4))])
def test_hierarchy_uses_first_layer_grid(fake_backbone, layers, grid):
    model = PatchCore(layers=layers, **options())
    images = batches(torch.randn(2, 3, 8, 8))
    model.fit(images)
    assert model.feature_grid == grid
    assert model.predict(images).patch_maps.shape == (2, *grid)


@pytest.mark.parametrize("kwargs", [{"scales": []}, {"scales": [1.5]},
                                    {"scales": [1., 0.]}, {"scales": [1., float("nan")]},
                                    {"fusion": "median"}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        MultiResPatchCore(device="cpu", **kwargs)


def test_unfitted_empty_and_one_shot_loaders(fake_backbone):
    model = MultiResPatchCore(**options())
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict([])
    with pytest.raises(ValueError, match="empty"):
        model.fit([])
    with pytest.raises(ValueError, match="re-iterable"):
        model.fit(iter(batches(torch.zeros(1, 3, 8, 8))))


def test_configs_keep_baseline_preprocessing_and_coreset():
    baseline = load_config("configs/patchcore_wrn50.yaml")
    for path in Path("configs").glob("patchcore_ms_*.yaml"):
        config = load_config(path)
        assert config["data"] == baseline["data"]
        assert config["model"]["coreset_ratio"] == 0.1
        assert config["scoring"]["sigma"] == 4.0


@pytest.mark.gpu
@pytest.mark.data
def test_bottle_twenty_image_gpu_smoke():
    root = os.environ.get("MVTEC_ROOT")
    if root is None or not (Path(root) / "bottle").is_dir():
        pytest.skip("MVTEC_ROOT/bottle is missing")
    train = DataLoader(MVTecDataset("bottle", "train", root=root, max_items=20),
                       batch_size=5, num_workers=0)
    # Predict on a separate normal validation split; no labels or masks enter models.
    validation = DataLoader(MVTecDataset("bottle", "val", root=root, max_items=5),
                            batch_size=5, num_workers=0)
    model = MultiResPatchCore(device="cuda", fusion="zmean")
    model.fit(train)
    before = copy.deepcopy(model.normalization)
    prediction = model.predict(validation)
    assert prediction.patch_maps.shape == (5, 28, 28)
    assert prediction.image_scores is None
    assert np.isfinite(prediction.patch_maps).all()
    assert model.normalization == before
    assert [m.feature_grid for m in model.models] == [(28, 28), (42, 42)]
    assert [len(m.memory_bank) for m in model.models] == [1568, 3528]
