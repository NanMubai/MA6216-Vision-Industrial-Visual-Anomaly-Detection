import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ad.base import build_model
from ad.data import MVTecDataset
from ad.patchcore import PatchCore


class ImageOnlyBatch(dict):
    """Fail immediately if a model attempts to inspect labels or masks."""

    def __getitem__(self, key):
        assert key == "image", f"model accessed forbidden batch field {key!r}"
        return super().__getitem__(key)


class FakeBackbone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, images):
        features = [images[:, :2] * self.scale]
        if len(self.layers) > 1:
            features.append(F.avg_pool2d(images * self.scale, 2))
        return features


@pytest.fixture
def fake_backbone(monkeypatch):
    def build(name, layers):
        assert name == "wrn50"
        return FakeBackbone(layers)

    monkeypatch.setattr("ad.patchcore.build_backbone", build)


def batches(images, size=2):
    return [ImageOnlyBatch(image=batch) for batch in images.split(size)]


def test_shapes_stats_frozen_backbone_and_no_label_access(fake_backbone):
    model = build_model("patchcore", device="cpu", target_dim=8, proj_dim=3,
                        coreset_ratio=0.25)
    images = torch.randn(3, 3, 4, 6)
    model.fit(batches(images))
    prediction = model.predict(batches(images))
    assert prediction.patch_maps.shape == (3, 4, 6)
    assert prediction.image_scores.shape == (3,)
    assert prediction.patch_maps.dtype == prediction.image_scores.dtype == np.float32
    assert np.isfinite(prediction.patch_maps).all()
    assert np.isfinite(prediction.image_scores).all()
    assert model.memory_bank.shape == (18, 8)
    assert model.memory_bank.dtype == torch.float32
    assert not model.memory_bank.requires_grad
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.stats() == {"memory_bank_size": 18, "feature_dim": 8,
                             "feature_grid": (4, 6)}


def test_embeddings_match_official_unfold_interpolate_mean_mapper(fake_backbone):
    model = PatchCore(device="cpu", patch_size=3, target_dim=11)
    images = torch.randn(2, 3, 4, 6, generator=torch.Generator().manual_seed(4))
    reference_layers = []
    with torch.no_grad():
        for feature in model.backbone(images):
            batch, channels, height, width = feature.shape
            unfolded = F.unfold(feature, kernel_size=3, padding=1)
            spatial = unfolded.reshape(batch, channels * 9, height, width)
            # The reference expands every channel/patch offset to the first
            # layer grid before flattening and applying the official MeanMapper.
            spatial = F.interpolate(spatial, size=(4, 6), mode="bilinear", align_corners=False)
            patches = spatial.permute(0, 2, 3, 1).reshape(-1, 1, channels * 9)
            reference_layers.append(F.adaptive_avg_pool1d(patches, 11).squeeze(1))
        stacked = torch.stack(reference_layers, dim=1).reshape(-1, 1, 22)
        expected = F.adaptive_avg_pool1d(stacked, 11).squeeze(1)
        actual, grid = model._embed(images)
    assert grid == (4, 6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_identical_features_have_zero_scores_across_chunks(fake_backbone):
    model = PatchCore(device="cpu", target_dim=16, coreset_ratio=1)
    model._QUERY_CHUNK_SIZE = 7
    model._BANK_CHUNK_SIZE = 9
    images = torch.randn(3, 3, 4, 4, generator=torch.Generator().manual_seed(2))
    model.fit(batches(images))
    prediction = model.predict(batches(images, size=3))
    np.testing.assert_allclose(prediction.patch_maps, 0, atol=1e-6)
    np.testing.assert_allclose(prediction.image_scores, 0, atol=1e-6)


def test_outlier_patch_has_maximum_score(fake_backbone):
    model = PatchCore(device="cpu", layers=("layer2",), patch_size=1,
                      target_dim=2, coreset_ratio=1)
    model.fit(batches(torch.zeros(2, 3, 4, 4)))
    images = torch.zeros(1, 3, 4, 4)
    images[0, :2, 1, 2] = 10
    prediction = model.predict(batches(images))
    assert np.unravel_index(prediction.patch_maps[0].argmax(), (4, 4)) == (1, 2)
    assert prediction.patch_maps[0, 1, 2] == pytest.approx(math.sqrt(200))
    assert prediction.image_scores[0] > 0
    assert np.count_nonzero(prediction.patch_maps) == 1


@pytest.mark.parametrize("neighbors", [1, 2, 3, 9])
def test_image_score_matches_equation_seven(fake_backbone, neighbors):
    model = PatchCore(device="cpu", layers=("layer2",), patch_size=1, target_dim=1,
                      coreset_ratio=1, num_neighbors=neighbors)
    # Both channels have the same value, so each patch embedding is the scalar
    # itself. m*=2 has support [2, 0, 5]; the maximum query patch is 3.5.
    train = torch.tensor([0., 2., 5.]).view(1, 1, 1, 3).expand(1, 3, 1, 3)
    test = torch.tensor([0., 3.5]).view(1, 1, 1, 2).expand(1, 3, 1, 2)
    model.fit(batches(train))
    # Use a common feature grid while retaining a unique max-distance query.
    test = torch.cat((test, torch.zeros(1, 3, 1, 1)), dim=-1)
    prediction = model.predict(batches(test))
    support_distances = torch.tensor([1.5, 3.5, 1.5])[:min(neighbors, 3)]
    weight = 1 - support_distances.softmax(dim=0)[0].item()
    assert prediction.patch_maps[0, 0, 1] == pytest.approx(1.5)
    assert prediction.image_scores[0] == pytest.approx(1.5 * weight)


def test_reweighting_is_stable_for_large_distances(fake_backbone):
    model = PatchCore(device="cpu", layers=("layer2",), patch_size=1, target_dim=1,
                      coreset_ratio=1, num_neighbors=3)
    train = torch.tensor([0., 2., 5.]).view(1, 1, 1, 3).expand(1, 3, 1, 3)
    model.fit(batches(train))
    prediction = model.predict(batches(torch.full((1, 3, 1, 3), 10000.)))
    expected = 9995 * (1 - torch.tensor([9995., 9998., 10000.]).softmax(dim=0)[0].item())
    assert np.isfinite(prediction.image_scores).all()
    assert prediction.image_scores[0] == pytest.approx(expected, rel=1e-6)


def test_chunked_neighbor_search_matches_dense_distances(fake_backbone):
    model = PatchCore(device="cpu", target_dim=5, coreset_ratio=1)
    model._QUERY_CHUNK_SIZE = 3
    model._BANK_CHUNK_SIZE = 4
    generator = torch.Generator().manual_seed(13)
    model.memory_bank = torch.randn(19, 5, generator=generator)
    queries = torch.randn(11, 5, generator=generator)
    dense = torch.cdist(queries, model.memory_bank)
    for count in (1, 3, 7):
        actual, indices = model._nearest_neighbors(queries, count=count)
        expected, expected_indices = dense.topk(count, dim=1, largest=False)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(indices, expected_indices)


def test_empty_loaders_and_unfitted_prediction(fake_backbone):
    model = PatchCore(device="cpu", target_dim=4, coreset_ratio=1)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict([])
    with pytest.raises(ValueError, match="empty"):
        model.fit([])
    model.fit(batches(torch.zeros(1, 3, 2, 2)))
    prediction = model.predict([])
    assert prediction.patch_maps.shape == (0, 2, 2)
    assert prediction.image_scores.shape == (0,)


@pytest.mark.gpu
@pytest.mark.data
def test_bottle_gpu_data_smoke():
    root = os.environ.get("MVTEC_ROOT")
    if root is None or not (Path(root) / "bottle").is_dir():
        pytest.skip("MVTEC_ROOT/bottle is missing")
    train = DataLoader(MVTecDataset("bottle", "train", root=root, max_items=20),
                       batch_size=5, num_workers=0)
    test = DataLoader(MVTecDataset("bottle", "test", root=root, max_items=10),
                      batch_size=5, num_workers=0)
    model = PatchCore(device="cuda")
    model.fit(train)
    prediction = model.predict(test)
    assert model.memory_bank.shape == (int(20 * 28 * 28 * 0.1), 1024)
    assert model.memory_bank.device.type == "cuda"
    assert model.memory_bank.dtype == torch.float32
    assert model.stats()["feature_grid"] == (28, 28)
    assert prediction.patch_maps.shape == (10, 28, 28)
    assert prediction.image_scores.shape == (10,)
    assert np.isfinite(prediction.patch_maps).all()
    assert np.isfinite(prediction.image_scores).all()
    assert (prediction.patch_maps >= 0).all()
    assert (prediction.image_scores >= 0).all()
