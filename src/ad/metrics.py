"""Evaluation metrics. Test labels/masks are used here and ONLY here.

Owner: branch feat/eval.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _binary_arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.shape != scores.shape:
        raise ValueError("labels/masks and scores/maps must have the same shape")
    return labels.ravel(), scores.ravel()


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _binary_arrays(labels, scores)
    return float(roc_auc_score(labels, scores))


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Binary prediction = scores >= threshold. Returns {f1, precision, recall, accuracy}."""
    labels, scores = _binary_arrays(labels, scores)
    predicted = scores >= threshold
    positive = labels != 0
    tp = int(np.count_nonzero(predicted & positive))
    predicted_count = int(np.count_nonzero(predicted))
    positive_count = int(np.count_nonzero(positive))
    return {
        "f1": _ratio(2 * tp, predicted_count + positive_count),
        "precision": _ratio(tp, predicted_count),
        "recall": _ratio(tp, positive_count),
        "accuracy": _ratio(int(np.count_nonzero(predicted == positive)), labels.size),
    }


def f1_max(labels: np.ndarray, scores: np.ndarray) -> dict:
    """ORACLE best F1 over all thresholds (uses test labels; report as reference only).
    Returns {f1, threshold}."""
    labels, scores = _binary_arrays(labels, scores)
    if not labels.size:
        raise ValueError("oracle F1 requires at least one sample")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # The last PR point has no threshold and must not participate in the argmax.
    denominator = precision[:-1] + recall[:-1]
    values = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(values))
    return {"f1": float(values[best]), "threshold": float(thresholds[best])}


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """masks (N,H,W) in {0,1}, maps (N,H,W) float. Over all pixels of all test images."""
    # Preserve float32 scores and byte labels even for tens of millions of pixels.
    masks, maps = _binary_arrays(masks, maps)
    return float(roc_auc_score(masks, maps))


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, num_thresholds: int = 1000) -> float:
    """Normalised area under the per-region-overlap curve up to ``fpr_limit``.

    For each threshold t: PRO(t) = mean over every connected GT region (8-connectivity,
    over all anomalous test images) of |region ∩ {map >= t}| / |region|;
    FPR(t) = fraction of normal pixels (mask == 0, all test images) with map >= t.
    Use ``num_thresholds`` thresholds (default 1000), evenly spaced from the minimum
    to the maximum map score, to reduce sensitivity to isolated extreme scores.
    Integrate PRO over FPR in [0, fpr_limit] (interpolate at the limit) and divide by
    ``fpr_limit`` so a perfect map scores 1.0. Return NaN when there are no GT regions.
    """
    masks = np.asarray(masks, dtype=np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    if masks.shape != maps.shape or maps.ndim != 3 or not maps.size:
        raise ValueError("masks and maps must have the same nonempty (N,H,W) shape")
    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be in (0, 1]")
    if not isinstance(num_thresholds, (int, np.integer)) or num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")
    minimum, maximum = float(maps.min()), float(maps.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("maps must contain finite scores")
    if not np.any(masks):
        return float("nan")
    if minimum == maximum:
        return 0.0

    thresholds = np.linspace(minimum, maximum, num_thresholds)
    normal_scores = np.sort(maps[masks == 0])
    if not normal_scores.size:
        raise ValueError("AU-PRO requires normal pixels to compute FPR")
    fpr = (normal_scores.size - np.searchsorted(normal_scores, thresholds, side="left"))
    fpr = fpr / normal_scores.size
    pro = np.zeros(num_thresholds, dtype=np.float64)
    region_count = 0
    for mask, score in zip(masks, maps):
        components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        # Bounding boxes avoid scanning an entire image once for each small defect.
        for region_id, bounds in enumerate(ndimage.find_objects(components, count), start=1):
            region_scores = np.sort(score[bounds][components[bounds] == region_id])
            detected = region_scores.size - np.searchsorted(region_scores, thresholds, side="left")
            pro += detected / region_scores.size
        region_count += count
    pro /= region_count

    # Descending thresholds give increasing FPR. Retain equal-FPR points: vertical
    # PRO jumps at FPR=0 are essential for a perfect map to integrate to one.
    fpr = np.concatenate(([0.0], fpr[::-1]))
    pro = np.concatenate(([0.0], pro[::-1]))
    stop = int(np.searchsorted(fpr, fpr_limit, side="right"))
    x, y = fpr[:stop], pro[:stop]
    if x[-1] < fpr_limit:
        at_limit = np.interp(fpr_limit, fpr[stop - 1:stop + 1], pro[stop - 1:stop + 1])
        x = np.append(x, fpr_limit)
        y = np.append(y, at_limit)
    return float(np.trapezoid(y, x) / fpr_limit)


def evaluate(labels: np.ndarray, image_scores: np.ndarray, masks: np.ndarray, maps: np.ndarray,
             image_thr: float, pixel_thr: float | None = None) -> dict:
    """All metrics in one dict: image_auroc, image_f1 (+precision/recall at image_thr),
    image_f1_max_oracle, pixel_auroc, aupro, and (if pixel_thr) pixel_f1 / pixel_iou."""
    image_metrics = f1_at_threshold(labels, image_scores, image_thr)
    result = {
        "image_auroc": image_auroc(labels, image_scores),
        **{f"image_{name}": value for name, value in image_metrics.items()},
        "image_f1_max_oracle": f1_max(labels, image_scores)["f1"],
        "pixel_auroc": pixel_auroc(masks, maps),
        "aupro": aupro(masks, maps),
    }
    if pixel_thr is not None:
        anomalous = np.asarray(labels) != 0
        masks = np.asarray(masks, dtype=np.uint8)[anomalous]
        maps = np.asarray(maps, dtype=np.float32)[anomalous]
        predicted = maps >= pixel_thr
        positive = masks != 0
        tp = int(np.count_nonzero(predicted & positive))
        predicted_count = int(np.count_nonzero(predicted))
        positive_count = int(np.count_nonzero(positive))
        result["pixel_f1"] = _ratio(2 * tp, predicted_count + positive_count)
        result["pixel_iou"] = _ratio(tp, predicted_count + positive_count - tp)
    return result
