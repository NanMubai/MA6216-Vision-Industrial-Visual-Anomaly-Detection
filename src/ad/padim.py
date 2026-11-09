"""PaDiM (Defard et al., ICPR 2021) - baseline.

Owner: branch feat/padim. Implements ad.base.AnomalyModel.
"""

from __future__ import annotations

from ad.base import AnomalyModel, Prediction


class PaDiM(AnomalyModel):
    def __init__(self, device: str = "cuda", backbone: str = "wrn50",
                 layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
                 d_reduced: int = 550, eps: float = 0.01, seed: int = 0):
        super().__init__(device)
        raise NotImplementedError

    def fit(self, loader) -> None:
        raise NotImplementedError

    def predict(self, loader) -> Prediction:
        raise NotImplementedError
