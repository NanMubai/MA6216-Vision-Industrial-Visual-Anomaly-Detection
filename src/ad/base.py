"""Shared model interface. Every anomaly detection method implements AnomalyModel.

Contract
--------
- ``fit(loader)`` sees ONLY normal training images (batches from MVTecDataset split="train").
  It must not read ``batch["label"]`` or ``batch["mask"]``.
- ``predict(loader)`` returns a ``Prediction`` with RAW patch-level anomaly maps at the
  model's native feature resolution (e.g. 28x28 for PatchCore-WRN50, 56x56 for PaDiM).
  Upsampling to image size, Gaussian smoothing and image-score aggregation are done
  afterwards by ``ad.scoring`` (shared by all methods so comparisons are fair).
- ``image_scores`` is optional: set it when the method defines its own image score
  (e.g. PatchCore's re-weighted max patch score). Otherwise leave None and
  ``ad.scoring`` derives image scores from the post-processed maps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class Prediction:
    patch_maps: np.ndarray                 # (N, h, w) float32, higher = more anomalous
    image_scores: np.ndarray | None = None  # (N,) float32 or None


class AnomalyModel(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def fit(self, loader: DataLoader) -> None: ...

    @abstractmethod
    def predict(self, loader: DataLoader) -> Prediction: ...

    def stats(self) -> dict:
        """Optional model statistics for reporting (memory-bank size, feature dims, ...)."""
        return {}


def build_model(method: str, device: str = "cuda", **kwargs) -> AnomalyModel:
    if method == "padim":
        from ad.padim import PaDiM
        return PaDiM(device=device, **kwargs)
    if method == "patchcore":
        from ad.patchcore import PatchCore
        return PatchCore(device=device, **kwargs)
    if method == "patchcore_multires":
        from ad.multiscale import MultiResPatchCore
        return MultiResPatchCore(device=device, **kwargs)
    raise ValueError(f"unknown method {method!r}")
