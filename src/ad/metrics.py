"""Evaluation metrics. Test labels/masks are used here and ONLY here.

Owner: branch feat/eval.
"""

from __future__ import annotations

import numpy as np


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    raise NotImplementedError


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Binary prediction = scores >= threshold. Returns {f1, precision, recall, accuracy}."""
    raise NotImplementedError


def f1_max(labels: np.ndarray, scores: np.ndarray) -> dict:
    """ORACLE best F1 over all thresholds (uses test labels; report as reference only).
    Returns {f1, threshold}."""
    raise NotImplementedError


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """masks (N,H,W) in {0,1}, maps (N,H,W) float. Over all pixels of all test images."""
    raise NotImplementedError


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, num_thresholds: int = 200) -> float:
    """Normalised area under the per-region-overlap curve up to ``fpr_limit``.

    For each threshold t: PRO(t) = mean over every connected GT region (8-connectivity,
    over all anomalous test images) of |region ∩ {map >= t}| / |region|;
    FPR(t) = fraction of normal pixels (mask == 0, all test images) with map >= t.
    Integrate PRO over FPR in [0, fpr_limit] (interpolate at the limit) and divide by
    ``fpr_limit`` so a perfect map scores 1.0.
    """
    raise NotImplementedError


def evaluate(labels: np.ndarray, image_scores: np.ndarray, masks: np.ndarray, maps: np.ndarray,
             image_thr: float, pixel_thr: float | None = None) -> dict:
    """All metrics in one dict: image_auroc, image_f1 (+precision/recall at image_thr),
    image_f1_max_oracle, pixel_auroc, aupro, and (if pixel_thr) pixel_f1 / pixel_iou."""
    raise NotImplementedError
