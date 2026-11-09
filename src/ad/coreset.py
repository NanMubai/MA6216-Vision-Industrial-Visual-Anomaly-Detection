"""Greedy k-center coreset subsampling (PatchCore, Roth et al. CVPR 2022).

Owner: branch feat/patchcore.
"""

from __future__ import annotations

import torch


def greedy_coreset(features: torch.Tensor, ratio: float = 0.1, proj_dim: int = 128,
                   seed: int = 0) -> torch.Tensor:
    """features (M,D) -> LongTensor of selected row indices, length max(1, int(M*ratio)).

    Johnson-Lindenstrauss random projection to ``proj_dim`` (if D > proj_dim), then
    iterative farthest-point selection with minimum-distance updates, on features.device.
    """
    raise NotImplementedError
