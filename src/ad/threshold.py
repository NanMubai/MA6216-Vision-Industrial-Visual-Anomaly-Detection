"""Decision thresholds computed ONLY from normal validation data (no test labels).

Owner: branch feat/eval.
"""

from __future__ import annotations

import numpy as np


def image_threshold(val_image_scores: np.ndarray, quantile: float = 0.99) -> float:
    """Image-level threshold = ``quantile`` of normal validation image scores."""
    raise NotImplementedError


def pixel_threshold(val_maps: np.ndarray, quantile: float = 0.999) -> float:
    """Pixel-level threshold = ``quantile`` of all normal validation pixel scores (N,H,W)."""
    raise NotImplementedError
