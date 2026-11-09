"""Frozen pretrained feature extractors.

WRN50 uses torchvision ImageNet weights (``wide_resnet50_2.tv_in1k``), the same weights as the
PaDiM / PatchCore papers. Weights are cached in ~/.cache/huggingface; run with
HF_HUB_OFFLINE=1 when there is no network access.
"""

from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F

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


DINOV2_MODELS = {
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
}
DINOV2_LAYERS = ("block5", "block8", "block11")


class DINOv2Features(nn.Module):
    """Frozen DINOv2 patch maps in requested order, with optional final LayerNorm.

    ``blockN`` names are zero-indexed. Input tensors already have ImageNet
    normalisation; bilinear resizing changes only their spatial size. timm's
    NCHW intermediate API removes *all* prefix tokens (CLS and any registers)
    before reshaping the patch tokens, including for register-equipped models.

    ``norm=True`` preserves the original a-priori experiment: apply the final
    learned LayerNorm to every selected block. ``norm=False`` returns raw
    block outputs; the final norm was trained for the last block and its affine
    scaling can substantially change intermediate-channel distance geometry.
    """

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 layers: tuple[str, ...] = DINOV2_LAYERS,
                 input_size: int = 448, pretrained: bool = True, norm: bool = True):
        super().__init__()
        self.layers = tuple(layers)
        if (not isinstance(input_size, int) or input_size < 14
                or input_size % 14):
            raise ValueError("input_size must be a positive multiple of 14")
        if not self.layers or any(
                not isinstance(layer, str) or not layer.startswith("block")
                or not layer[5:].isdigit() for layer in self.layers):
            raise ValueError("DINOv2 layers must be nonempty zero-indexed blockN names")
        indices = [int(layer[5:]) for layer in self.layers]
        self.input_size = input_size
        self.norm = norm
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       img_size=input_size)
        if any(index >= len(self.model.blocks) for index in indices):
            raise ValueError(f"DINOv2 block index must be below {len(self.model.blocks)}")
        self._indices = sorted(set(indices))
        self._reorder = [self._indices.index(index) for index in indices]
        self.model.eval().requires_grad_(False)

    @property
    def channels(self) -> list[int]:
        return [self.model.num_features] * len(self.layers)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        features = self.model.forward_intermediates(
            x, indices=self._indices, norm=self.norm, return_prefix_tokens=False,
            output_fmt="NCHW", intermediates_only=True, stop_early=True)
        return [features[index] for index in self._reorder]


class WRN50DINOv2Concat(nn.Module):
    """WRN50 layer2+3 followed by DINOv2 blocks, on one bilinear grid.

    ``layers`` selects DINOv2 blocks; the WRN50 branch always uses layer2+3.
    The default 28x28 grid preserves the baseline PatchCore patch count.
    Maps remain separate so PatchCore's unchanged per-layer MeanMapper and
    Aggregator perform the feature fusion. Neither branch is reweighted.
    ``norm`` is passed only to DINOv2; the default final-LayerNorm scaling is
    retained for the a-priori combined variant, so branch scales are not matched.
    """

    def __init__(self, layers: tuple[str, ...] = DINOV2_LAYERS,
                 input_size: int = 448, grid_size: int = 28,
                 pretrained: bool = True, norm: bool = True):
        super().__init__()
        if not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError("grid_size must be a positive integer")
        self.grid_size = grid_size
        self.wrn50 = WRN50Features(layers=("layer2", "layer3"), pretrained=pretrained)
        self.dinov2 = DINOv2Features(layers=layers, input_size=input_size,
                                    pretrained=pretrained, norm=norm)

    @property
    def channels(self) -> list[int]:
        return self.wrn50.channels + self.dinov2.channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.wrn50(x) + self.dinov2(x)
        return [F.interpolate(feature, size=(self.grid_size, self.grid_size),
                              mode="bilinear", align_corners=False)
                for feature in features]


def build_backbone(name: str = "wrn50", **kwargs) -> nn.Module:
    """Build frozen features; ``dinov2_vitb14_raw`` always disables final LayerNorm.

    The raw name is for post-hoc, test-informed sensitivity analysis; existing
    DINOv2 names retain their original ``norm=True`` default.
    """
    if name == "wrn50":
        return WRN50Features(**kwargs)
    if name == "dinov2_vitb14_raw":
        return DINOv2Features(model_name=DINOV2_MODELS["dinov2_vitb14"],
                               **{**kwargs, "norm": False})
    if name in DINOV2_MODELS:
        return DINOv2Features(model_name=DINOV2_MODELS[name], **kwargs)
    if name == "wrn50+dinov2_vitb14":
        return WRN50DINOv2Concat(**kwargs)
    raise ValueError(f"unknown backbone {name!r}")
