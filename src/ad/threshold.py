"""Decision thresholds computed ONLY from normal validation data (no test labels).

Owner: branch feat/eval.
"""

from __future__ import annotations

import numpy as np


def image_threshold(val_image_scores: np.ndarray, quantile: float = 0.99) -> float:
    """Image-level threshold = ``quantile`` of normal validation image scores."""
    scores = np.asarray(val_image_scores)
    if not scores.size:
        raise ValueError("normal validation image scores must not be empty")
    return float(np.quantile(scores, quantile))


def pixel_threshold(val_maps: np.ndarray, quantile: float = 0.999) -> float:
    """Pixel-level threshold = ``quantile`` of all normal validation pixel scores (N,H,W)."""
    maps = np.asarray(val_maps)
    if not maps.size:
        raise ValueError("normal validation maps must not be empty")
    # NumPy uses partitioning for a scalar quantile, avoiding a full pixel sort.
    # Keep overwrite_input=False (the default) so calibration never mutates maps.
    return float(np.quantile(maps, quantile))


def global_image_threshold(val_scores_by_category: dict[str, np.ndarray],
                           mode: str = "z3", quantile: float = 0.99) -> float:
    """One threshold for all categories, using normalised normal validation ONLY.

    ``z3`` is fixed a priori at 3. ``pooled_quantile`` pools images (rather than
    giving each category equal weight) and takes their empirical quantile.
    This function never accepts test labels, masks, or test scores.
    """
    if mode == "z3":
        return 3.0
    if mode != "pooled_quantile":
        raise ValueError(f"unknown global threshold mode: {mode!r}")
    if not val_scores_by_category:
        raise ValueError("normal validation scores from all categories are required")
    values = [np.asarray(scores, dtype=np.float32).ravel()
              for scores in val_scores_by_category.values()]
    if any(not scores.size or not np.isfinite(scores).all() for scores in values):
        raise ValueError("each category needs finite normal validation scores")
    return image_threshold(np.concatenate(values), quantile)
