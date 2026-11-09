"""Greedy k-center coreset subsampling (PatchCore, Roth et al. CVPR 2022).

Owner: branch feat/patchcore.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, ratio: float = 0.1, proj_dim: int = 128,
                   seed: int = 0) -> torch.Tensor:
    """features (M,D) -> LongTensor of selected row indices, length max(1, int(M*ratio)).

    Johnson-Lindenstrauss random projection to ``proj_dim`` (if D > proj_dim), then
    iterative farthest-point selection with minimum-distance updates, on features.device.
    """
    if features.ndim != 2 or min(features.shape) == 0:
        raise ValueError("features must be a nonempty (M, D) tensor")
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    if not isinstance(proj_dim, int) or proj_dim < 1:
        raise ValueError("proj_dim must be a positive integer")

    size, dim = features.shape
    count = max(1, int(size * ratio))
    if count == size:
        return torch.arange(size, device=features.device)

    # A local generator leaves the application's RNG state untouched. Keep every
    # index on-device, including the argmax, to avoid a CUDA sync per iteration.
    generator = torch.Generator(device=features.device).manual_seed(seed)
    index = torch.randint(size, (1,), generator=generator, device=features.device)
    working = features.detach().to(dtype=torch.float32)
    if dim > proj_dim:
        projection = torch.randn(dim, proj_dim, generator=generator,
                                 device=features.device) / math.sqrt(proj_dim)
        working = working @ projection
    working = working.contiguous()

    squared_norms = working.square().sum(dim=1)
    min_distances = torch.full((size,), torch.inf, device=features.device)
    selected = torch.empty(count, dtype=torch.long, device=features.device)
    for i in range(count):
        selected[i:i + 1].copy_(index)
        center = working.index_select(0, index).squeeze(0)
        # Squared distances have the same farthest-point ordering as distances;
        # matrix-vector multiplication avoids an M x D temporary per iteration.
        distances = torch.mv(working, center).mul_(-2)
        distances.add_(squared_norms).add_(squared_norms.index_select(0, index))
        distances.clamp_min_(0)
        torch.minimum(min_distances, distances, out=min_distances)
        # Exclude selected rows explicitly, including when features are duplicates.
        min_distances.index_fill_(0, index, -1)
        index = min_distances.argmax().reshape(1)

    return selected
