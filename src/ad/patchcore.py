"""PatchCore (Roth et al., CVPR 2022) - main method.

Owner: branch feat/patchcore. Implements ad.base.AnomalyModel.
"""

from __future__ import annotations

from ad.base import AnomalyModel, Prediction


class PatchCore(AnomalyModel):
    def __init__(self, device: str = "cuda", backbone: str = "wrn50",
                 layers: tuple[str, ...] = ("layer2", "layer3"),
                 patch_size: int = 3, target_dim: int = 1024,
                 coreset_ratio: float = 0.1, proj_dim: int = 128,
                 num_neighbors: int = 9, seed: int = 0):
        super().__init__(device)
        raise NotImplementedError

    def fit(self, loader) -> None:
        raise NotImplementedError

    def predict(self, loader) -> Prediction:
        raise NotImplementedError
