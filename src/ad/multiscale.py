"""PatchCore input-pyramid fusion, with training-only optional scale calibration.

The zmean reference is the coreset-to-full-training-bank nearest-neighbour
Euclidean distance distribution, excluding each selected patch's own source row.
It is patch leave-one-out, not image leave-one-out: other patches from the same
normal training image remain eligible. Validation is used only by ad.threshold.
Image scores are delegated to ad.scoring (max after common post-processing),
which differs from single-scale PatchCore's Eq. 7 image-score reweighting.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from ad.base import AnomalyModel, Prediction
from ad.coreset import greedy_coreset
from ad.patchcore import PatchCore


class _CalibratedPatchCore(PatchCore):
    """Reuse PatchCore embeddings/search and retain full features only during fit."""

    @torch.inference_mode()
    def fit(self, loader) -> None:
        self.backbone.eval()
        self.normalization = None
        features, grid = [], None
        for batch in loader:
            embedded, batch_grid = self._embed(batch["image"])
            if grid is not None and batch_grid != grid:
                raise ValueError("all training images must have the same feature grid")
            grid = batch_grid
            if len(embedded):
                features.append(embedded)
        if not features:
            raise ValueError("cannot fit PatchCore on an empty loader")
        full_bank = torch.cat(features)
        features.clear()
        if len(full_bank) < 2:
            raise ValueError("zmean requires at least two training patches")
        indices = greedy_coreset(full_bank, ratio=self.coreset_ratio,
                                 proj_dim=self.proj_dim, seed=self.seed)
        coreset = full_bank.index_select(0, indices).float().contiguous()
        # Search the FULL training bank, excluding each coreset patch's source
        # row. Including self would make every calibration distance zero.
        self.memory_bank = full_bank
        try:
            distances, _ = self._nearest_neighbors(coreset, exclude_indices=indices)
            reference = distances[:, 0].double()
            mean, std = reference.mean().item(), reference.std(correction=0).item()
            self.normalization = (mean, std if std > 0 else 1.0)
            self.reference_count = len(reference)
        finally:
            self.memory_bank = coreset
        self.feature_grid = grid


class _ResizedBatches:
    """Expose only image tensors, resized on the model device inside the model."""

    def __init__(self, loader, scale: float, device: str):
        self.loader, self.scale, self.device = loader, scale, device

    def __iter__(self):
        for batch in self.loader:
            images = batch["image"].to(device=self.device, dtype=torch.float32)
            if self.scale != 1.0:
                size = tuple(max(1, int(length * self.scale)) for length in images.shape[-2:])
                images = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
            yield {"image": images}


class MultiResPatchCore(AnomalyModel):
    """Independent PatchCore banks at each input scale; fuse at the 1.0 grid.

    ``mean`` averages raw maps. ``zmean`` averages maps standardized with frozen
    fit-only coreset-to-full-bank patch-distance statistics. Both return
    ``image_scores=None`` so the shared scoring pipeline derives a max score.
    The loader passed to fit must be re-iterable, as a DataLoader is.
    """

    def __init__(self, device: str = "cuda", scales=(1.0, 1.5), fusion: str = "mean",
                 **patchcore_options):
        super().__init__(device)
        self.scales = tuple(float(scale) for scale in scales)
        if not self.scales or any(not math.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("scales must be nonempty, finite and positive")
        if 1.0 not in self.scales:
            raise ValueError("scales must include the base scale 1.0")
        if fusion not in ("mean", "zmean"):
            raise ValueError("fusion must be 'mean' or 'zmean'")
        self.fusion = fusion
        self.base_index = self.scales.index(1.0)
        model_class = _CalibratedPatchCore if fusion == "zmean" else PatchCore
        self.models = [model_class(device=device, **patchcore_options) for _ in self.scales]
        self.normalization: tuple[tuple[float, float], ...] = ()
        self.fit_seconds_per_scale: list[float] = []
        self.infer_ms_per_image_per_scale: list[float] = []
        self._fitted = False

    def _synchronize(self):
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def fit(self, loader) -> None:
        if isinstance(loader, Iterator):
            raise ValueError("fit requires a re-iterable loader, not a one-shot iterator")
        self._fitted = False
        self.normalization = ()
        self.fit_seconds_per_scale = []
        self.infer_ms_per_image_per_scale = []
        for scale, model in zip(self.scales, self.models):
            self._synchronize()
            started = time.perf_counter()
            model.fit(_ResizedBatches(loader, scale, self.device))
            self._synchronize()
            self.fit_seconds_per_scale.append(time.perf_counter() - started)
        if self.fusion == "zmean":
            self.normalization = tuple(model.normalization for model in self.models)
        else:
            self.normalization = tuple((0.0, 1.0) for _ in self.models)
        self._fitted = True

    @torch.inference_mode()
    def predict(self, loader) -> Prediction:
        if not self._fitted:
            raise RuntimeError("MultiResPatchCore must be fitted before prediction")
        base_grid = self.models[self.base_index].feature_grid
        fused_maps = []
        scale_seconds = [0.0 for _ in self.models]
        image_count = 0
        for batch in loader:
            aligned = []
            for index, (scale, model, (mean, std)) in enumerate(
                    zip(self.scales, self.models, self.normalization)):
                self._synchronize()
                started = time.perf_counter()
                prediction = model.predict(_ResizedBatches([batch], scale, self.device))
                self._synchronize()
                scale_seconds[index] += time.perf_counter() - started
                maps = torch.from_numpy(prediction.patch_maps).to(self.device)
                # Normalise on the native grid before aligning scale maps.
                maps = (maps - mean) / std
                if tuple(maps.shape[-2:]) != base_grid:
                    maps = F.interpolate(maps[:, None], size=base_grid, mode="bilinear",
                                         align_corners=False)[:, 0]
                aligned.append(maps)
            fused = torch.stack(aligned).mean(dim=0)
            fused_maps.append(fused.cpu().numpy())
            image_count += len(fused)
        self.infer_ms_per_image_per_scale = [
            1000 * seconds / image_count if image_count else 0.0 for seconds in scale_seconds]
        maps = (np.concatenate(fused_maps) if fused_maps
                else np.empty((0, *base_grid), dtype=np.float32))
        return Prediction(maps, image_scores=None)

    def stats(self) -> dict:
        scales = []
        for index, (scale, model) in enumerate(zip(self.scales, self.models)):
            bank = model.memory_bank
            scales.append({
                "scale": scale, **model.stats(),
                "memory_bank_bytes": 0 if bank is None else bank.numel() * bank.element_size(),
                "fit_seconds": (self.fit_seconds_per_scale[index]
                                if index < len(self.fit_seconds_per_scale) else None),
                "infer_ms_per_image": (self.infer_ms_per_image_per_scale[index]
                                       if self.infer_ms_per_image_per_scale else None),
                "normalization": (self.normalization[index] if self.normalization else None),
                "normalization_reference_count": getattr(model, "reference_count", 0),
            })
        return {"fusion": self.fusion, "scales": scales,
                "image_score": "shared_max", "normalization_reference": (
                    "coreset_to_full_train_bank_self_excluded" if self.fusion == "zmean" else None)}
