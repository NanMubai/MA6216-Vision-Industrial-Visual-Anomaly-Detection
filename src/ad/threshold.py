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
