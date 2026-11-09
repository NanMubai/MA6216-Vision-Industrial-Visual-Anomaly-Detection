import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from ad.scoring import image_scores_from_maps, upsample_and_smooth, zscore_normalize


@pytest.fixture(autouse=True)
def scoring_device_and_threads(monkeypatch, request):
    # Unit tests use CPU without the shared GPU lock; the marked test checks CUDA.
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _scipy_reference(patches, size, sigma):
    upsampled = F.interpolate(torch.from_numpy(patches)[:, None], size=(size, size),
                              mode="bilinear", align_corners=False)[:, 0].numpy()
    if sigma > 0:
        return np.stack([gaussian_filter(m, sigma=sigma, mode="reflect", truncate=4.0)
                         for m in upsampled])
    return upsampled


@pytest.mark.parametrize("size,sigma", [(23, 4.0), (13, 2.7), (3, 5.0), (1, 1.1), (7, 0.1)])
def test_smoothing_matches_scipy_reflect(size, sigma):
    patches = np.random.default_rng(5).normal(size=(2, 3, 5)).astype(np.float32) * 100
    actual = upsample_and_smooth(patches, size, sigma)
    expected = _scipy_reference(patches, size, sigma)
    assert actual.dtype == np.float32
    assert actual.shape == (2, size, size)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_disabled_smoothing_is_plain_bilinear_upsampling(sigma):
    patches = np.array([[[0.0, 2.0], [4.0, 8.0]]], dtype=np.float32)
    np.testing.assert_array_equal(upsample_and_smooth(patches, 5, sigma),
                                  _scipy_reference(patches, 5, sigma))


def test_scoring_batches_and_noncontiguous_inputs():
    patches = np.random.default_rng(8).random((35, 3, 5), dtype=np.float32)[:, :, ::-1]
    actual = upsample_and_smooth(patches, 11, sigma=1.5)
    expected = _scipy_reference(patches.copy(), 11, sigma=1.5)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    assert upsample_and_smooth(np.empty((0, 3, 5), dtype=np.float32), 11).shape == (0, 11, 11)


def test_cuda_failure_falls_back_to_cpu_without_allocating_gpu(monkeypatch):
    arange = torch.arange
    devices = []

    def unavailable_cuda(*args, **kwargs):
        device = kwargs.get("device")
        devices.append(device.type)
        if device.type == "cuda":
            raise RuntimeError("CUDA device unavailable")
        return arange(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "arange", unavailable_cuda)
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    actual = upsample_and_smooth(patches, 9, sigma=1.0)
    np.testing.assert_allclose(actual, _scipy_reference(patches, 9, 1.0), atol=1e-4, rtol=0)
    assert devices[0] == "cuda" and set(devices[1:]) == {"cpu"}


@pytest.mark.gpu
def test_cuda_smoothing_matches_scipy(monkeypatch):
    import ad.scoring as scoring

    devices = []
    smooth = scoring._smooth_batch

    def record_device(batch, kernel):
        devices.append(batch.device.type)
        return smooth(batch, kernel)

    monkeypatch.setattr(scoring, "_smooth_batch", record_device)
    patches = np.random.default_rng(7).normal(size=(35, 3, 5)).astype(np.float32) * 100
    actual = upsample_and_smooth(patches, 23, sigma=4.0)
    np.testing.assert_allclose(actual, _scipy_reference(patches, 23, 4.0), atol=1e-4, rtol=0)
    assert devices == ["cuda", "cuda"]


def test_image_score_aggregation():
    maps = np.array([[[1, 2], [3, 4]], [[8, 6], [4, 2]]], dtype=np.float32)
    np.testing.assert_array_equal(image_scores_from_maps(maps), [4, 8])
    np.testing.assert_array_equal(image_scores_from_maps(maps, "topk_mean", 0.5), [3.5, 7])
    np.testing.assert_array_equal(image_scores_from_maps(maps, "topk_mean", 0.01), [4, 8])
    np.testing.assert_array_equal(image_scores_from_maps(maps, "topk_mean", 1), [2.5, 5])
    assert image_scores_from_maps(maps, "topk_mean").dtype == np.float32


def test_zscore_uses_only_normal_validation_reference():
    reference = np.array([1, 3], dtype=np.float32)
    scores = np.array([-1, 1, 3, 5], dtype=np.float32)
    np.testing.assert_array_equal(zscore_normalize(scores, reference), [-3, -1, 1, 3])
    np.testing.assert_array_equal(reference, [1, 3])
    np.testing.assert_array_equal(zscore_normalize(scores, np.array([2, 2])), scores - 2)
    assert zscore_normalize(scores, reference).dtype == np.float32


def test_invalid_scoring_inputs():
    maps = np.ones((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        upsample_and_smooth(np.ones((3, 4)), 8)
    with pytest.raises(ValueError, match="size"):
        upsample_and_smooth(maps, 0)
    with pytest.raises(ValueError, match="mode"):
        image_scores_from_maps(maps, "invalid")
    for ratio in (0, -0.1, 1.1):
        with pytest.raises(ValueError, match="topk_ratio"):
            image_scores_from_maps(maps, "topk_mean", ratio)
    with pytest.raises(ValueError, match="ref"):
        zscore_normalize(np.ones(2), np.array([]))
