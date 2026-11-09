"""Frozen pretrained feature extractors.

WRN50 uses torchvision ImageNet weights (``wide_resnet50_2.tv_in1k``), the same weights as the
PaDiM / PatchCore papers. Weights are cached in ~/.cache/huggingface; run with
HF_HUB_OFFLINE=1 when there is no network access.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

WRN50_LAYER_INDEX = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
# feature channels / stride for WRN50 at each layer
WRN50_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
WRN50_STRIDE = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}


class WRN50Features(nn.Module):
    """Returns a list of feature maps, one per requested layer, in the order given."""

    def __init__(self, layers: tuple[str, ...] = ("layer2", "layer3"), pretrained: bool = True):
        super().__init__()
        self.layers = tuple(layers)
        idx = [WRN50_LAYER_INDEX[l] for l in self.layers]
        self.model = timm.create_model("wide_resnet50_2.tv_in1k", pretrained=pretrained,
                                       features_only=True, out_indices=idx)
        self.model.eval().requires_grad_(False)
        order = sorted(range(len(idx)), key=lambda i: idx[i])
        self._reorder = [order.index(i) for i in range(len(idx))]  # timm returns ascending order

    @property
    def channels(self) -> list[int]:
        return [WRN50_CHANNELS[l] for l in self.layers]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.model(x)
        return [feats[i] for i in self._reorder]


def build_backbone(name: str = "wrn50", **kwargs) -> nn.Module:
    if name == "wrn50":
        return WRN50Features(**kwargs)
    raise ValueError(f"unknown backbone {name!r} (DINOv2 is added on branch exp/dinov2)")
