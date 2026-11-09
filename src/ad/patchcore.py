"""PatchCore (Roth et al., CVPR 2022) - main method.

Owner: branch feat/patchcore. Implements ad.base.AnomalyModel.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ad.backbones import build_backbone
from ad.base import AnomalyModel, Prediction
from ad.coreset import greedy_coreset


class PatchCore(AnomalyModel):
    # Bound the temporary distance matrix independently of loader and bank sizes.
    _QUERY_CHUNK_SIZE = 1024
    _BANK_CHUNK_SIZE = 4096

    def __init__(self, device: str = "cuda", backbone: str = "wrn50",
                 layers: tuple[str, ...] = ("layer2", "layer3"),
                 patch_size: int = 3, target_dim: int = 1024,
                 coreset_ratio: float = 0.1, proj_dim: int = 128,
                 num_neighbors: int = 9, seed: int = 0):
        super().__init__(device)
        if not layers:
            raise ValueError("at least one backbone layer is required")
        if not isinstance(patch_size, int) or patch_size < 1 or patch_size % 2 != 1:
            raise ValueError("patch_size must be a positive odd integer")
        if not isinstance(target_dim, int) or target_dim < 1:
            raise ValueError("target_dim must be a positive integer")
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")
        if not isinstance(proj_dim, int) or proj_dim < 1:
            raise ValueError("proj_dim must be a positive integer")
        if not isinstance(num_neighbors, int) or num_neighbors < 1:
            raise ValueError("num_neighbors must be a positive integer")

        self.layers = tuple(layers)
        self.patch_size, self.target_dim = patch_size, target_dim
        self.coreset_ratio, self.proj_dim = coreset_ratio, proj_dim
        self.num_neighbors, self.seed = num_neighbors, seed
        self.backbone = build_backbone(backbone, layers=self.layers).to(device)
        self.backbone.eval().requires_grad_(False)
        self.memory_bank: torch.Tensor | None = None
        self.feature_grid: tuple[int, int] | None = None

    def _embed(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Official unfold -> per-layer MeanMapper -> Aggregator embeddings."""
        features = self.backbone(images.to(device=self.device, dtype=torch.float32))
        grid = tuple(features[0].shape[-2:])
        layers = []
        for feature in features:
            batch, _, height, width = feature.shape
            patches = F.unfold(feature, kernel_size=self.patch_size, stride=1,
                               padding=(self.patch_size - 1) // 2)
            # Each patch is flattened in (channel, patch_y, patch_x) order, as in
            # the official preprocessing MeanMapper, rather than channel pooling.
            reduced = F.adaptive_avg_pool1d(patches.transpose(1, 2), self.target_dim)
            reduced = reduced.transpose(1, 2).reshape(batch, self.target_dim, height, width)
            if (height, width) != grid:
                # Both operations are linear: reducing before interpolation is
                # equivalent to interpolating every unfolded channel first, with
                # substantially smaller intermediate tensors for deep layers.
                reduced = F.interpolate(reduced, size=grid, mode="bilinear",
                                        align_corners=False)
            layers.append(reduced.flatten(2).transpose(1, 2))

        stacked = torch.stack(layers, dim=2).flatten(2)
        embeddings = F.adaptive_avg_pool1d(stacked, self.target_dim)
        return embeddings.reshape(-1, self.target_dim).contiguous(), grid

    @torch.inference_mode()
    def fit(self, loader) -> None:
        self.backbone.eval()
        features = []
        grid = None
        for batch in loader:
            embeddings, batch_grid = self._embed(batch["image"])
            if grid is not None and batch_grid != grid:
                raise ValueError("all training images must have the same feature grid")
            grid = batch_grid
            if len(embeddings):
                features.append(embeddings)
        if not features:
            raise ValueError("cannot fit PatchCore on an empty loader")

        all_features = torch.cat(features, dim=0)
        features.clear()
        indices = greedy_coreset(all_features, ratio=self.coreset_ratio,
                                 proj_dim=self.proj_dim, seed=self.seed)
        self.memory_bank = all_features.index_select(0, indices).float().contiguous()
        self.feature_grid = grid

    def _nearest_neighbors(self, queries: torch.Tensor, count: int = 1,
                           exclude_indices: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact Euclidean search, chunking both query and memory-bank axes."""
        distance_chunks, index_chunks = [], []
        for start in range(0, len(queries), self._QUERY_CHUNK_SIZE):
            query = queries[start:start + self._QUERY_CHUNK_SIZE]
            best_distances = query.new_full((len(query), count), torch.inf)
            best_indices = torch.zeros((len(query), count), device=query.device,
                                       dtype=torch.long)
            for bank_start in range(0, len(self.memory_bank), self._BANK_CHUNK_SIZE):
                bank = self.memory_bank[bank_start:bank_start + self._BANK_CHUNK_SIZE]
                distances = torch.cdist(query, bank, p=2)
                if exclude_indices is not None:
                    bank_indices = torch.arange(bank_start, bank_start + len(bank),
                                                device=query.device)
                    excluded = exclude_indices[start:start + len(query), None]
                    distances.masked_fill_(bank_indices[None, :] == excluded, torch.inf)
                if count == 1:
                    values, indices = distances.min(dim=1, keepdim=True)
                    improve = values < best_distances
                    best_distances = torch.minimum(best_distances, values)
                    best_indices = torch.where(improve, indices + bank_start, best_indices)
                else:
                    values, indices = distances.topk(min(count, len(bank)), dim=1,
                                                     largest=False)
                    candidates = torch.cat((best_distances, values), dim=1)
                    candidate_indices = torch.cat((best_indices, indices + bank_start), dim=1)
                    best_distances, positions = candidates.topk(count, dim=1, largest=False)
                    best_indices = candidate_indices.gather(1, positions)
            # cdist's matrix-multiplication path can leave positive round-off for
            # identical vectors. Recompute only the selected distances directly.
            distances = torch.linalg.vector_norm(
                query[:, None, :] - self.memory_bank[best_indices], dim=-1)
            distance_chunks.append(distances)
            index_chunks.append(best_indices)
        return torch.cat(distance_chunks), torch.cat(index_chunks)

    @torch.inference_mode()
    def predict(self, loader) -> Prediction:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore must be fitted before prediction")
        self.backbone.eval()
        patch_maps, image_scores = [], []
        for batch in loader:
            images = batch["image"]
            features, grid = self._embed(images)
            if grid != self.feature_grid:
                raise ValueError("prediction feature grid must match the training grid")
            if not len(features):
                continue
            distances, nearest = self._nearest_neighbors(features)
            maps = distances.reshape(len(images), *grid)
            scores, patch_indices = distances.reshape(len(images), -1).max(dim=1)
            max_indices = patch_indices + torch.arange(len(images), device=features.device) * (
                grid[0] * grid[1])
            max_features = features[max_indices]
            anchors = nearest[max_indices, 0]

            count = min(self.num_neighbors, len(self.memory_bank))
            support = anchors[:, None]
            if count > 1:
                _, other_neighbors = self._nearest_neighbors(
                    self.memory_bank[anchors], count=count - 1, exclude_indices=anchors)
                support = torch.cat((support, other_neighbors), dim=1)
            support_distances = torch.linalg.vector_norm(
                max_features[:, None, :] - self.memory_bank[support], dim=-1)
            # Eq. 7: m* is explicitly the first support element. Softmax avoids
            # overflow in exp(distance), including for very anomalous patches.
            weights = 1 - torch.softmax(support_distances, dim=1)[:, 0]
            patch_maps.append(maps.cpu().numpy())
            image_scores.append((weights * scores).cpu().numpy())

        if not patch_maps:
            return Prediction(np.empty((0, *self.feature_grid), dtype=np.float32),
                              np.empty(0, dtype=np.float32))
        return Prediction(np.concatenate(patch_maps), np.concatenate(image_scores))

    def stats(self) -> dict:
        return {
            "memory_bank_size": 0 if self.memory_bank is None else len(self.memory_bank),
            "feature_dim": self.target_dim,
            "feature_grid": self.feature_grid,
        }
