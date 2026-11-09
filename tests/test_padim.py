import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ad import padim
from ad.base import Prediction
from ad.data import MVTecDataset, get_root


class FakeBackbone(nn.Module):
    """Two resolutions with distinct values, including a trainable parameter."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, images):
        assert not self.training
        assert not torch.is_grad_enabled()
        return [images[:, :2] * self.scale, images[:, 2:, ::2, ::2] * self.scale]


class ImagesOnly(dict):
    def __getitem__(self, key):
        if key != "image":
            raise AssertionError(f"model accessed forbidden batch field {key}")
        return super().__getitem__(key)


def batches(images, batch_size):
    return [ImagesOnly(image=part) for part in images.split(batch_size)]


@pytest.fixture
def fake_backbone(monkeypatch):
    calls = []

    def build(name, **kwargs):
        calls.append((name, kwargs))
        return FakeBackbone()

    monkeypatch.setattr(padim, "build_backbone", build)
    # Exercise several chunks and an incomplete final chunk on tiny images.
    monkeypatch.setattr(padim.PaDiM, "_position_chunk_size", 5)
    return calls


def reference_embedding(images, indices):
    values = images.numpy().astype(np.float64)
    low = values[:, 2:, ::2, ::2].repeat(2, axis=2).repeat(2, axis=3)
    return np.concatenate([values[:, :2], low], axis=1)[:, indices]


@pytest.mark.parametrize("d_reduced,batch_size", [(1, 1), (2, 4), (3, 11)])
def test_mahalanobis_matches_numpy(fake_backbone, d_reduced, batch_size):
    rng = torch.Generator().manual_seed(45)
    train = torch.randn(11, 3, 4, 6, generator=rng)
    test = torch.randn(5, 3, 4, 6, generator=rng)
    model = padim.PaDiM(device="cpu", d_reduced=d_reduced, eps=0.03, seed=13)
    model.fit(iter(batches(train, batch_size)))
    prediction = model.predict(iter(batches(test, 2)))
    indices = model.channel_indices.numpy()
    train_features = reference_embedding(train, indices)
    test_features = reference_embedding(test, indices)
    expected = np.empty((5, 4, 6))
    for row in range(4):
        for col in range(6):
            values = train_features[:, :, row, col]
            covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=False))
            covariance += model.eps * np.eye(d_reduced)
            delta = test_features[:, :, row, col] - values.mean(axis=0)
            expected[:, row, col] = np.sqrt(np.einsum(
                "ni,ij,nj->n", delta, np.linalg.inv(covariance), delta))
    np.testing.assert_allclose(prediction.patch_maps, expected, rtol=2e-6, atol=2e-6)
    assert isinstance(prediction, Prediction)
    assert prediction.patch_maps.shape == (5, 4, 6)
    assert prediction.patch_maps.dtype == np.float32
    assert prediction.image_scores is None
    assert model.mean.device.type == model.inv_covariance.device.type == "cpu"
    assert model.mean.dtype == model.inv_covariance.dtype == torch.float32
    assert model.stats() == {
        "feature_dim": d_reduced, "full_feature_dim": 3,
        "map_size": (4, 6), "fit_image_count": 11,
    }
    assert fake_backbone == [("wrn50", {"layers": ("layer1", "layer2", "layer3")})]
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_seed_and_refit_are_deterministic(fake_backbone):
    images = torch.randn(9, 3, 4, 6, generator=torch.Generator().manual_seed(12))
    model = padim.PaDiM(device="cpu", d_reduced=2, seed=7)
    rng_state = torch.random.get_rng_state()
    model.fit(batches(images, 4))
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    expected_indices = torch.randperm(3, generator=torch.Generator().manual_seed(7))[:2]
    assert torch.equal(model.channel_indices, expected_indices)
    original = model.predict(batches(images, 3)).patch_maps
    # Re-fitting must replace statistics, not add another copy of the data.
    model.fit(batches(images, 2))
    np.testing.assert_allclose(model.predict(batches(images, 4)).patch_maps, original, rtol=2e-6)
    assert model.stats()["fit_image_count"] == 9


def test_regularized_rank_deficient_covariance_and_empty_prediction(fake_backbone):
    images = torch.ones(3, 3, 4, 6)
    model = padim.PaDiM(device="cpu", d_reduced=3, eps=0.25)
    model.fit(batches(images, 2))
    np.testing.assert_array_equal(model.predict(batches(images, 2)).patch_maps, 0)
    expected = np.sqrt(3 / model.eps)
    np.testing.assert_allclose(model.predict(batches(images + 1, 2)).patch_maps, expected)
    prediction = model.predict([])
    assert prediction.patch_maps.shape == (0, 4, 6)
    assert prediction.patch_maps.dtype == np.float32
    assert prediction.image_scores is None


def test_invalid_training_data_and_prediction_state(fake_backbone):
    model = padim.PaDiM(device="cpu", d_reduced=2)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict([])
    for images in (torch.empty(0, 3, 4, 6), torch.ones(1, 3, 4, 6)):
        with pytest.raises(ValueError, match="at least two"):
            model.fit(batches(images, 2))
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict([])
    model.fit(batches(torch.ones(2, 3, 4, 6), 2))
    with pytest.raises(ValueError, match="map size changed"):
        model.predict(batches(torch.ones(2, 3, 6, 6), 2))
    with pytest.raises(ValueError, match="exceeds feature dimension"):
        padim.PaDiM(device="cpu", d_reduced=4).fit(batches(torch.ones(2, 3, 4, 6), 2))


@pytest.mark.parametrize("kwargs", [
    {"d_reduced": 0}, {"d_reduced": 1.5}, {"eps": 0},
    {"eps": -1}, {"eps": float("nan")}, {"layers": ()},
])
def test_invalid_configuration(fake_backbone, kwargs):
    with pytest.raises(ValueError):
        padim.PaDiM(device="cpu", **kwargs)


@pytest.mark.gpu
@pytest.mark.data
def test_bottle_smoke():
    root = get_root()
    if not (root / "bottle").is_dir():
        pytest.skip(f"MVTec AD bottle data missing: {root / 'bottle'}")
    train = MVTecDataset("bottle", "train", root=root, max_items=20)
    test = MVTecDataset("bottle", "test", root=root)
    # The dataset orders defect folders before good; an initial slice would
    # contain no normal examples. Choose five from each group, without tuning.
    good = [i for i, sample in enumerate(test.samples) if sample[3] == "good"][:5]
    defect = [i for i, sample in enumerate(test.samples) if sample[3] != "good"][:5]
    assert len(train) == 20 and len(good) == len(defect) == 5
    model = padim.PaDiM(device="cuda")
    torch.cuda.reset_peak_memory_stats()
    model.fit(DataLoader(train, batch_size=4, num_workers=0))
    prediction = model.predict(DataLoader(Subset(test, good + defect), batch_size=2, num_workers=0))
    assert prediction.patch_maps.shape == (10, 56, 56)
    assert prediction.patch_maps.dtype == np.float32
    assert prediction.image_scores is None
    assert np.isfinite(prediction.patch_maps).all()
    scores = prediction.patch_maps.max(axis=(1, 2))
    assert scores[5:].mean() > scores[:5].mean()
    assert torch.cuda.max_memory_allocated() < 12 * 1024**3
