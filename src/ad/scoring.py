"""Shared post-processing: patch maps -> pixel anomaly maps -> image scores.

Owner: branch feat/eval. Signatures are fixed contracts; keep them.
"""

from __future__ import annotations

import numpy as np


def upsample_and_smooth(patch_maps: np.ndarray, size: int, sigma: float = 4.0) -> np.ndarray:
    """(N,h,w) raw patch maps -> (N,size,size) float32 maps.

    Bilinear upsampling (align_corners=False), then per-image Gaussian blur with std ``sigma``
    pixels (sigma <= 0 disables smoothing). Use a truncate of 4*sigma like scipy's default.
    """
    raise NotImplementedError


def image_scores_from_maps(maps: np.ndarray, mode: str = "max", topk_ratio: float = 0.01) -> np.ndarray:
    """(N,H,W) maps -> (N,) image scores. mode: "max" | "topk_mean" (mean of top ratio pixels)."""
    raise NotImplementedError


def zscore_normalize(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalise ``scores`` with mean/std of ``ref`` (normal validation scores only)."""
    raise NotImplementedError
