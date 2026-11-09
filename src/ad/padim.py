"""PaDiM (Defard et al., ICPR 2021) with one Gaussian per feature position.

Reduced embeddings are cached on CPU in float32. Training accumulates float64
sums and outer products one position chunk at a time, with sample covariance
normalisation (ddof=1). Prediction transfers each inverse covariance chunk only
once, reusing it across all image batches. Fitted statistics reside on CPU in
float32, pinned when using CUDA.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ad.backbones import build_backbone
from ad.base import AnomalyModel, Prediction


class PaDiM(AnomalyModel):
    # At d=550 each float64 matrix chunk occupies about 148 MiB. Keeping only
    # one chunk on the device avoids a 7.1 GiB full covariance accumulator.
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

    def _collect_embeddings(self, loader) -> list[torch.Tensor]:
        """Consume a loader once, caching (positions, batch, channels) on CPU."""
        embeddings = []
        for batch in loader:
            images = batch["image"]
            if images.shape[0] == 0:
                continue
            embedding = self._embed(images)
            if self.map_size is None:
                self.map_size = tuple(embedding.shape[-2:])
            # Contiguous position slices can later be transferred without
            # gathering strided channels or retaining embeddings on the GPU.
            embeddings.append(embedding.flatten(2).permute(2, 0, 1).to(
                device="cpu", memory_format=torch.contiguous_format))
        return embeddings

    @torch.no_grad()
    def fit(self, loader) -> None:
        """Fit normal training images only; never inspect labels or masks."""
        self.backbone.eval()
        self.mean = self.inv_covariance = None
        self.map_size = None
        self._fit_image_count = 0
        embeddings = self._collect_embeddings(loader)
        image_count = sum(embedding.shape[1] for embedding in embeddings)
        if image_count < 2:
            raise ValueError("PaDiM needs at least two training images for sample covariance")

        positions = math.prod(self.map_size)
        use_cuda = torch.device(self.device).type == "cuda"
        means = torch.empty(positions, self.d_reduced, dtype=torch.float32,
                            device="cpu", pin_memory=use_cuda)
        inverse = torch.empty(positions, self.d_reduced, self.d_reduced,
                              dtype=torch.float32, device="cpu", pin_memory=use_cuda)
        for start in range(0, positions, self._position_chunk_size):
            stop = min(start + self._position_chunk_size, positions)
            sums = torch.zeros(stop - start, self.d_reduced,
                               dtype=torch.float64, device=self.device)
            covariance = torch.zeros(stop - start, self.d_reduced, self.d_reduced,
                                     dtype=torch.float64, device=self.device)
            for embedding in embeddings:
                values = embedding[start:stop].to(device=self.device, dtype=torch.float64)
                sums.add_(values.sum(dim=1))
                covariance.baddbmm_(values.transpose(1, 2), values)
                del values
            mean = sums.div_(image_count).unsqueeze(2)
            # E[xx^T] - E[x]E[x]^T, using N-1 for unbiased sample covariance.
            covariance.baddbmm_(mean, mean.transpose(1, 2), alpha=-image_count)
            covariance.div_(image_count - 1)
            covariance.diagonal(dim1=-2, dim2=-1).add_(self.eps)
            cholesky = torch.linalg.cholesky(covariance)
            inverse[start:stop].copy_(torch.cholesky_inverse(cholesky))
            means[start:stop].copy_(mean.squeeze(2))
            del sums, covariance, mean, cholesky

        self.mean = means
        self.inv_covariance = inverse
        self._fit_image_count = image_count
        if use_cuda:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def predict(self, loader) -> Prediction:
        if self.mean is None or self.inv_covariance is None:
            raise RuntimeError("PaDiM must be fitted before prediction")
        self.backbone.eval()
        embeddings = self._collect_embeddings(loader)
        image_count = sum(embedding.shape[1] for embedding in embeddings)
        positions = math.prod(self.map_size)
        distances = torch.empty(image_count, positions, dtype=torch.float32, device="cpu")
        if image_count == 0:
            return Prediction(patch_maps=distances.reshape(0, *self.map_size).numpy())
        for start in range(0, positions, self._position_chunk_size):
            stop = start + self._position_chunk_size
            mean = self.mean[start:stop].to(self.device, non_blocking=True).unsqueeze(1)
            inverse = self.inv_covariance[start:stop].to(self.device, non_blocking=True)
            image_start = 0
            for embedding in embeddings:
                delta = embedding[start:stop].to(self.device) - mean
                squared = (torch.bmm(delta, inverse) * delta).sum(dim=2)
                image_stop = image_start + embedding.shape[1]
                distances[image_start:image_stop, start:stop].copy_(
                    squared.clamp_min_(0).sqrt_().transpose(0, 1))
                image_start = image_stop
        patch_maps = distances.reshape(image_count, *self.map_size).numpy()
        return Prediction(patch_maps=patch_maps, image_scores=None)

    def stats(self) -> dict:
        return {
            "feature_dim": self.d_reduced,
            "full_feature_dim": self._full_feature_dim,
            "map_size": self.map_size,
            "fit_image_count": self._fit_image_count,
        }
