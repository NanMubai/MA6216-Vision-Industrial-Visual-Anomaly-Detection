"""Shared post-processing: patch maps -> pixel anomaly maps -> image scores.

Owner: branch feat/eval. Signatures are fixed contracts; keep them.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F


def _smooth_batch(maps: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable blur with SciPy's half-sample symmetric (``reflect``) edges."""
    radius = kernel.numel() // 2
    for axis in (2, 3):
        length = maps.shape[axis]
        indices = torch.arange(-radius, length + radius, device=maps.device) % (2 * length)
        indices = torch.minimum(indices, 2 * length - 1 - indices)
        padded = maps.index_select(axis, indices)
        weight = kernel.view(1, 1, -1, 1) if axis == 2 else kernel.view(1, 1, 1, -1)
        maps = F.conv2d(padded, weight)
    return maps


def upsample_and_smooth(patch_maps: np.ndarray, size: int, sigma: float = 4.0,
                        *, device: str | None = None) -> np.ndarray:
    """(N,h,w) raw patch maps -> (N,size,size) float32 maps.

    Bilinear upsampling (align_corners=False), then per-image Gaussian blur with std ``sigma``
    pixels (sigma <= 0 disables smoothing). Use a truncate of 4*sigma like scipy's default.
    ``device=None`` selects CUDA if available, falling back to CPU on CUDA failure.
    An explicit device is honoured; ``device="cpu"`` avoids probing or initialising CUDA.
    """
    patch_maps = np.asarray(patch_maps, dtype=np.float32)
    if patch_maps.ndim != 3 or min(patch_maps.shape[1:]) < 1:
        raise ValueError("patch_maps must have shape (N,h,w) with positive h and w")
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")

    result = np.empty((len(patch_maps), size, size), dtype=np.float32)
    if not len(patch_maps):
        return result
    radius = int(4.0 * sigma + 0.5) if sigma > 0 else 0

    def process(target_device: torch.device) -> np.ndarray:
        kernel = None
        if radius:
            positions = torch.arange(-radius, radius + 1, dtype=torch.float64, device=target_device)
            kernel = torch.exp(-0.5 * (positions / sigma).square())
            kernel = (kernel / kernel.sum()).to(torch.float32)
        # Bound intermediate storage independently of the number of test images.
        cudnn_flags = (torch.backends.cudnn.flags(allow_tf32=False)
                       if target_device.type == "cuda" else nullcontext())
        with torch.inference_mode(), cudnn_flags:
            for start in range(0, len(patch_maps), 32):
                batch = torch.from_numpy(np.ascontiguousarray(patch_maps[start:start + 32]))
                batch = F.interpolate(batch[:, None].to(target_device), size=(size, size),
                                      mode="bilinear", align_corners=False)
                if kernel is not None:
                    batch = _smooth_batch(batch, kernel)
                result[start:start + len(batch)] = batch[:, 0].cpu().numpy()
        return result

    if device is not None:
        return process(torch.device(device))
    if torch.cuda.is_available():
        try:
            return process(torch.device("cuda"))
        except RuntimeError:
            # A present but unavailable/exhausted CUDA device must not prevent scoring.
            pass
    return process(torch.device("cpu"))


def image_scores_from_maps(maps: np.ndarray, mode: str = "max", topk_ratio: float = 0.01) -> np.ndarray:
    """(N,H,W) maps -> (N,) image scores. mode: "max" | "topk_mean" (mean of top ratio pixels)."""
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or min(maps.shape[1:]) < 1:
        raise ValueError("maps must have shape (N,H,W) with positive H and W")
    pixels = maps.reshape(len(maps), maps.shape[1] * maps.shape[2])
    if mode == "max":
        return pixels.max(axis=1)
    if mode != "topk_mean":
        raise ValueError(f"unknown image score mode: {mode!r}")
    if not 0 < topk_ratio <= 1:
        raise ValueError("topk_ratio must be in (0, 1]")
    count = max(1, int(pixels.shape[1] * topk_ratio))
    top = np.partition(pixels, pixels.shape[1] - count, axis=1)[:, -count:]
    return top.mean(axis=1, dtype=np.float32)


def zscore_normalize(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalise ``scores`` with mean/std of ``ref`` (normal validation scores only)."""
    scores = np.asarray(scores, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if not ref.size:
        raise ValueError("ref must contain normal validation scores")
    mean = ref.mean(dtype=np.float64)
    std = ref.std(dtype=np.float64)
    # A constant validation reference has no scale; use a unit scale to stay finite.
    return ((scores.astype(np.float64) - mean) / (std if std > 0 else 1.0)).astype(np.float32)
