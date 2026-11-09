import json

import numpy as np
import pytest

from ad.metrics import aupro, evaluate, f1_at_threshold, f1_max, image_auroc, pixel_auroc
from ad.threshold import image_threshold, pixel_threshold


@pytest.mark.parametrize("metric", [image_auroc, pixel_auroc])
def test_auroc_perfect_inverted_and_random(metric):
    rng = np.random.default_rng(12)
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 5000)
    scores = rng.random(labels.size, dtype=np.float32)
    if metric is pixel_auroc:
        labels, scores = labels.reshape(10, 20, 50), scores.reshape(10, 20, 50)
    assert metric(labels, labels.astype(np.float32)) == 1.0
    assert metric(labels, 1.0 - labels) == 0.0
    assert metric(labels, scores) == pytest.approx(0.5, abs=0.05)


def test_f1_at_threshold_includes_equal_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.6, 0.9], dtype=np.float32)
    result = f1_at_threshold(labels, scores, float(scores[1]))
    assert result == {"f1": 0.5, "precision": 0.5, "recall": 0.5, "accuracy": 0.5}
    assert f1_at_threshold(labels, scores, 1.0) == {
        "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.5,
    }


def test_f1_max_is_oracle_over_actual_thresholds():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.6, 0.9], dtype=np.float32)
    result = f1_max(labels, scores)
    assert result["f1"] == pytest.approx(0.8)
    assert result["threshold"] == pytest.approx(0.6)
    assert f1_at_threshold(labels, scores, result["threshold"])["f1"] == result["f1"]
    assert f1_max(labels, np.ones(4))["f1"] == pytest.approx(2 / 3)


def test_f1_max_oracle_keeps_tied_scores_in_one_prediction_group():
    labels = np.array([1, 0, 1, 0], dtype=np.uint8)
    scores = np.array([0.9, 0.9, 0.4, 0.1], dtype=np.float32)
    # The tied .9 scores include one normal sample and cannot be separated.
    # At .4 there are two true positives and one false positive: F1 = 4/5.
    result = f1_max(labels, scores)
    assert result["threshold"] == pytest.approx(0.4)
    assert result["f1"] == pytest.approx(0.8)
    assert f1_at_threshold(labels, scores, result["threshold"])["f1"] == result["f1"]


def _region_masks():
    masks = np.zeros((3, 64, 64), dtype=np.uint8)
    masks[0, 3:13, 5:17] = 1
    masks[0, 30:45, 35:50] = 1
    masks[1, 8:34, 15:44] = 1
    return masks


@pytest.mark.parametrize("fpr_limit", [0.1, 0.3, 1.0])
def test_aupro_perfect(fpr_limit):
    masks = _region_masks()
    assert aupro(masks, masks.astype(np.float32), fpr_limit=fpr_limit) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("constant", [0.0, 3.0, -2.0])
def test_aupro_constant_has_no_localisation(constant):
    masks = _region_masks()
    assert aupro(masks, np.full(masks.shape, constant, dtype=np.float32)) == 0.0


@pytest.mark.parametrize("constant", [None, 0.0, 3.0])
def test_aupro_without_gt_regions_is_nan(constant):
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    maps = (np.random.default_rng(0).random(masks.shape, dtype=np.float32)
            if constant is None else np.full(masks.shape, constant, dtype=np.float32))
    result = aupro(masks, maps)
    assert type(result) is float and np.isnan(result)


def test_aupro_default_resolution_handles_an_outlier_pixel():
    masks = _region_masks()
    maps = np.random.default_rng(42).random(masks.shape, dtype=np.float32)
    maps[masks != 0] += 0.5
    maps[2, 0, 0] = 50.0
    reference = aupro(masks, maps, num_thresholds=10000)
    result = aupro(masks, maps)
    assert result == aupro(masks, maps, num_thresholds=1000)
    assert result == pytest.approx(reference, abs=1e-3)


def test_aupro_random_maps_follow_chance_curve():
    masks = _region_masks()
    maps = np.random.default_rng(24).random(masks.shape, dtype=np.float32)
    for limit in (0.1, 0.3, 0.5):
        assert aupro(masks, maps, fpr_limit=limit) == pytest.approx(limit / 2, abs=0.035)


def test_aupro_weights_regions_equally():
    masks = np.zeros((1, 100, 100), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    masks[0, 20:80, 20:80] = 1
    maps = np.random.default_rng(0).random(masks.shape, dtype=np.float32)
    maps[masks != 0] = 0.0
    maps[0, 1:3, 1:3] = 1.0
    assert aupro(masks, maps) == pytest.approx(0.5, abs=1e-3)


def test_aupro_uses_eight_connectivity():
    masks = np.zeros((1, 50, 50), dtype=np.uint8)
    masks[0, 1, 1] = 1
    masks[0, 2:4, 2:4] = 1  # Diagonally joins the one-pixel part into one region.
    maps = np.random.default_rng(1).random(masks.shape, dtype=np.float32)
    maps[masks != 0] = 0.0
    maps[0, 1, 1] = 1.0
    assert aupro(masks, maps) == pytest.approx(0.2, abs=1e-3)


def test_aupro_interpolates_at_fpr_limit():
    masks = np.array([[[1, 1, 0, 0]]], dtype=np.uint8)
    maps = np.array([[[1.0, 0.0, 0.8, 0.2]]], dtype=np.float32)
    # Three thresholds: 1 -> (FPR=0, PRO=.5), .5 -> (.5,.5), 0 -> (1,1).
    # Up to FPR=.75: .5*.5 + .25*(.5+.75)/2, then divide by .75.
    assert aupro(masks, maps, fpr_limit=0.75, num_thresholds=3) == pytest.approx(13 / 24)


def test_evaluate_pixel_metrics_only_use_anomalous_images():
    labels = np.array([0, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.9], dtype=np.float32)
    masks = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8)
    maps = np.array([[[0.8, 0.8], [0.8, 0.8]], [[1.0, 0.8], [0.1, 0.1]]], dtype=np.float32)
    result = evaluate(labels, scores, masks, maps, image_thr=0.5, pixel_thr=0.5)
    assert result["image_auroc"] == result["image_f1"] == result["image_f1_max_oracle"] == 1.0
    assert result["image_precision"] == result["image_recall"] == 1.0
    assert result["pixel_f1"] == pytest.approx(2 / 3)
    assert result["pixel_iou"] == 0.5
    assert all(type(value) is float for value in result.values())
    json.dumps(result, allow_nan=False)
    without_pixels = evaluate(labels, scores, masks, maps, image_thr=0.5)
    assert "pixel_f1" not in without_pixels and "pixel_iou" not in without_pixels
    assert evaluate(labels, scores, masks, maps, image_thr=0.5, pixel_thr=0.0)["pixel_f1"] == 0.4


@pytest.mark.parametrize("quantile,expected", [(0.0, 0.0), (0.5, 4.5), (0.9, 8.1), (1.0, 9.0)])
def test_thresholds_are_normal_validation_quantiles(quantile, expected):
    values = np.arange(10, dtype=np.float32)
    original = values.copy()
    assert image_threshold(values, quantile) == pytest.approx(expected)
    assert pixel_threshold(values.reshape(1, 2, 5), quantile) == pytest.approx(expected)
    np.testing.assert_array_equal(values, original)


def test_pixel_threshold_on_realistic_validation_size():
    values = np.arange(20 * 224 * 224, dtype=np.float32).reshape(20, 224, 224)
    assert pixel_threshold(values) == pytest.approx((values.size - 1) * 0.999)


def test_invalid_metric_and_calibration_inputs():
    with pytest.raises(ValueError, match="same shape"):
        pixel_auroc(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="fpr_limit"):
        aupro(_region_masks(), _region_masks(), fpr_limit=0)
    with pytest.raises(ValueError, match="num_thresholds"):
        aupro(_region_masks(), _region_masks(), num_thresholds=1)
    for threshold in (image_threshold, pixel_threshold):
        with pytest.raises(ValueError, match="empty"):
            threshold(np.array([]))
