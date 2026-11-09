"""PaDiM (Defard et al., ICPR 2021) with one Gaussian per feature position.

Training uses streaming float64 sums and outer products, with sample covariance
normalisation (ddof=1). Only position chunks are inverted or transferred during
prediction; the fitted means and inverse covariances reside on CPU in float32.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from ad.backbones import build_backbone
from ad.base import AnomalyModel, Prediction


class PaDiM(AnomalyModel):
    # At d=550 each float64 matrix chunk occupies about 148 MiB. The full
    # float64 accumulator at 56x56 occupies 7.1 GiB, leaving room for WRN50.
    _position_chunk_size = 64

    def __init__(self, device: str = "cuda", backbone: str = "wrn50",
                 layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
                 d_reduced: int = 550, eps: float = 0.01, seed: int = 0):
        super().__init__(device)
        if not isinstance(d_reduced, int) or d_reduced < 1:
            raise ValueError("d_reduced must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not layers:
            raise ValueError("layers must not be empty")
        self.layers = tuple(layers)
        self.d_reduced, self.eps, self.seed = d_reduced, eps, seed
        self.backbone = build_backbone(backbone, layers=self.layers)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.channel_indices: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None  # (positions, d), CPU float32
        self.inv_covariance: torch.Tensor | None = None  # (positions, d, d), CPU float32
        self.map_size: tuple[int, int] | None = None
        self._full_feature_dim: int | None = None
        self._fit_image_count = 0

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.to(self.device))
        map_size = tuple(features[0].shape[-2:])
        embedding = torch.cat([
            F.interpolate(feature, size=map_size, mode="nearest")
            if feature.shape[-2:] != map_size else feature
            for feature in features
        ], dim=1)
        full_dim = embedding.shape[1]
        if self.channel_indices is None:
            if self.d_reduced > full_dim:
                raise ValueError(f"d_reduced={self.d_reduced} exceeds feature dimension {full_dim}")
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self.channel_indices = torch.randperm(full_dim, generator=generator)[:self.d_reduced]
            self._full_feature_dim = full_dim
        elif full_dim != self._full_feature_dim:
            raise ValueError("backbone feature dimension changed after channel selection")
        if self.map_size is not None and map_size != self.map_size:
            raise ValueError(f"feature map size changed: expected {self.map_size}, got {map_size}")
        return embedding.index_select(1, self.channel_indices.to(embedding.device)).float()

    @torch.no_grad()
    def fit(self, loader) -> None:
        """Fit normal training images only; never inspect labels or masks."""
        self.backbone.eval()
        self.mean = self.inv_covariance = None
        self.map_size = None
        self._fit_image_count = 0
        sums = outer_products = None
        image_count = 0
        for batch in loader:
            images = batch["image"]
            if images.shape[0] == 0:
                continue
            embedding = self._embed(images)
            if sums is None:
                self.map_size = tuple(embedding.shape[-2:])
                positions = math.prod(self.map_size)
                sums = torch.zeros(positions, self.d_reduced, dtype=torch.float64,
                                   device=self.device)
                outer_products = torch.zeros(positions, self.d_reduced, self.d_reduced,
                                             dtype=torch.float64, device=self.device)
            embedding = embedding.flatten(2)  # (batch, channels, positions)
            for start in range(0, sums.shape[0], self._position_chunk_size):
                stop = start + self._position_chunk_size
                values = embedding[:, :, start:stop].permute(2, 1, 0).double()
                sums[start:stop].add_(values.sum(dim=2))
                # In-place accumulation avoids another full (positions,d,d) tensor.
                outer_products[start:stop].baddbmm_(values, values.transpose(1, 2))
                del values
            image_count += embedding.shape[0]
            del embedding

        if image_count < 2:
            raise ValueError("PaDiM needs at least two training images for sample covariance")

        means = sums.div_(image_count)
        inverse = torch.empty(outer_products.shape, dtype=torch.float32, device="cpu")
        for start in range(0, means.shape[0], self._position_chunk_size):
            stop = start + self._position_chunk_size
            mean = means[start:stop].unsqueeze(2)
            # E[xx^T] - E[x]E[x]^T, using N-1 for unbiased sample covariance.
            covariance = outer_products[start:stop]
            covariance.baddbmm_(mean, mean.transpose(1, 2), alpha=-image_count)
            covariance.div_(image_count - 1)
            covariance.diagonal(dim1=-2, dim2=-1).add_(self.eps)
            cholesky = torch.linalg.cholesky(covariance)
            inverse[start:stop].copy_(torch.cholesky_inverse(cholesky).to(
                device="cpu", dtype=torch.float32))

        self.mean = means.to(device="cpu", dtype=torch.float32)
        self.inv_covariance = inverse
        self._fit_image_count = image_count

    @torch.no_grad()
    def predict(self, loader) -> Prediction:
        if self.mean is None or self.inv_covariance is None:
            raise RuntimeError("PaDiM must be fitted before prediction")
        self.backbone.eval()
        maps = []
        for batch in loader:
            images = batch["image"]
            if images.shape[0] == 0:
                continue
            embedding = self._embed(images).flatten(2).permute(2, 0, 1)
            distances = torch.empty(embedding.shape[:2], device=self.device, dtype=torch.float32)
            for start in range(0, embedding.shape[0], self._position_chunk_size):
                stop = start + self._position_chunk_size
                delta = embedding[start:stop] - self.mean[start:stop].to(self.device).unsqueeze(1)
                inverse = self.inv_covariance[start:stop].to(self.device)
                squared = (torch.bmm(delta, inverse) * delta).sum(dim=2)
                distances[start:stop] = squared.clamp_min_(0).sqrt_()
            maps.append(distances.transpose(0, 1).reshape(-1, *self.map_size).cpu().numpy())
        patch_maps = np.concatenate(maps) if maps else np.empty((0, *self.map_size), dtype=np.float32)
        return Prediction(patch_maps=patch_maps, image_scores=None)

    def stats(self) -> dict:
        return {
            "feature_dim": self.d_reduced,
            "full_feature_dim": self._full_feature_dim,
            "map_size": self.map_size,
            "fit_image_count": self._fit_image_count,
        }
