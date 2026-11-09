"""DINOv2 interface, prefix-token exclusion, and unchanged PatchCore integration."""

import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import ad.backbones as backbones
from ad.backbones import DINOv2Features, WRN50DINOv2Concat, WRN50Features, build_backbone
from ad.config import load_config
from ad.data import MVTecDataset
from ad.patchcore import PatchCore


@pytest.mark.parametrize("name,constructor,extra", [
    ("wrn50", "WRN50Features", {}),
    ("dinov2_vitb14", "DINOv2Features",
     {"model_name": "vit_base_patch14_dinov2.lvd142m"}),
    ("dinov2_vitb14_raw", "DINOv2Features",
     {"model_name": "vit_base_patch14_dinov2.lvd142m", "norm": False}),
    ("dinov2_vits14", "DINOv2Features",
     {"model_name": "vit_small_patch14_dinov2.lvd142m"}),
    ("wrn50+dinov2_vitb14", "WRN50DINOv2Concat", {}),
])
def test_build_backbone_dispatch(monkeypatch, name, constructor, extra):
    factory = Mock()
    monkeypatch.setattr(backbones, constructor, factory)
    layers = ("block5", "block8", "block11")
    assert build_backbone(name, layers=layers, pretrained=False) is factory.return_value
    factory.assert_called_once_with(layers=layers, pretrained=False, **extra)


def test_unknown_backbone():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("not_a_backbone")


@pytest.mark.parametrize("kwargs", [
    {"input_size": 447}, {"input_size": 0}, {"layers": ()},
    {"layers": ("layer2",)}, {"layers": ("block-1",)},
])
def test_invalid_dinov2_options_fail_before_loading_weights(monkeypatch, kwargs):
    factory = Mock()
    monkeypatch.setattr(backbones.timm, "create_model", factory)
    with pytest.raises(ValueError):
        DINOv2Features(**kwargs)
    factory.assert_not_called()


def test_variant_configs_only_change_feature_backbone():
    baseline = load_config("configs/patchcore_wrn50.yaml")
    for name in ("patchcore_dinov2_vitb14", "patchcore_dinov2_vits14",
                 "patchcore_wrn50_dinov2"):
        config = load_config(f"configs/{name}.yaml")
        assert config["model"]["layers"] == ["block5", "block8", "block11"]
        for key in ("name",):
            config.pop(key)
        config["model"]["backbone"] = baseline["model"]["backbone"]
        config["model"]["layers"] = baseline["model"]["layers"]
        assert config == {key: value for key, value in baseline.items() if key != "name"}


@pytest.mark.gpu
@pytest.mark.parametrize("name,channels", [
    ("dinov2_vitb14", 768), ("dinov2_vits14", 384),
])
def test_dinov2_shapes_channels_order_and_cls_exclusion(name, channels):
    model = build_backbone(name, layers=("block11", "block5", "block8")).cuda()
    images = torch.randn(1, 3, 224, 224, device="cuda")
    output = model(images)
    assert model.input_size == 448
    assert model.channels == [channels] * 3
    assert [tuple(feature.shape) for feature in output] == [(1, channels, 32, 32)] * 3
    assert not model.model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    with torch.inference_mode():
        resized = F.interpolate(images, (448, 448), mode="bilinear", align_corners=False)
        # Independently request sequences and prefix tokens to check spatial
        # values, normalisation, order, and exclusion rather than only shape.
        sequences = model.model.forward_intermediates(
            resized, indices=[5, 8, 11], norm=True, return_prefix_tokens=True,
            output_fmt="NLC", intermediates_only=True)
        for feature, (patches, prefix) in zip(output, [sequences[2], sequences[0], sequences[1]]):
            assert patches.shape == (1, 1024, channels)
            assert prefix.shape == (1, model.model.num_prefix_tokens, channels)
            torch.testing.assert_close(feature.flatten(2).transpose(1, 2), patches)
            assert torch.isfinite(feature).all()


@pytest.mark.gpu
def test_dinov2_register_tokens_excluded():
    # Random weights avoid downloading an uncached register-equipped checkpoint.
    model = DINOv2Features(model_name="vit_small_patch14_reg4_dinov2",
                           layers=("block5",), pretrained=False).cuda()
    assert model.model.num_prefix_tokens == 5  # one CLS plus four registers
    output = model(torch.randn(1, 3, 224, 224, device="cuda"))
    assert output[0].shape == (1, 384, 32, 32)
    assert model.channels == [384]


@pytest.mark.gpu
def test_wrn50_dinov2_common_grid_and_channels():
    model = WRN50DINOv2Concat().cuda()
    images = torch.randn(1, 3, 224, 224, device="cuda")
    output = model(images)
    assert model.channels == [512, 1024, 768, 768, 768]
    assert [tuple(feature.shape) for feature in output] == [
        (1, channels, 28, 28) for channels in model.channels]
    assert all(torch.isfinite(feature).all() for feature in output)
    wrn = WRN50Features(layers=("layer2", "layer3")).cuda()
    reference = wrn(images)
    for actual, original in zip(output[:2], reference):
        expected = F.interpolate(original, (28, 28), mode="bilinear", align_corners=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.data
def test_patchcore_dinov2_bottle_smoke():
    root = os.environ.get("MVTEC_ROOT")
    if root is None or not (Path(root) / "bottle").is_dir():
        pytest.skip("MVTEC_ROOT/bottle is missing")
    train = DataLoader(MVTecDataset("bottle", "train", root=root, max_items=20),
                       batch_size=5, num_workers=0)
    test = DataLoader(MVTecDataset("bottle", "test", root=root, max_items=10),
                      batch_size=5, num_workers=0)
    model = PatchCore(device="cuda", backbone="dinov2_vitb14",
                      layers=("block5", "block8", "block11"))
    model.fit(train)
    prediction = model.predict(test)
    assert model.memory_bank.shape == (int(20 * 32 * 32 * 0.1), 1024)
    assert model.stats()["feature_grid"] == (32, 32)
    assert prediction.patch_maps.shape == (10, 32, 32)
    assert prediction.image_scores.shape == (10,)
    assert np.isfinite(prediction.patch_maps).all()
    assert np.isfinite(prediction.image_scores).all()
    assert (prediction.patch_maps >= 0).all()
    assert (prediction.image_scores >= 0).all()


def test_posthoc_config_only_disables_final_layernorm():
    baseline = load_config("configs/patchcore_dinov2_vitb14.yaml")
    posthoc = load_config("configs/patchcore_dinov2_vitb14_noln.yaml")
    assert posthoc["model"]["backbone"] == "dinov2_vitb14_raw"
    posthoc["name"] = baseline["name"]
    posthoc["model"]["backbone"] = baseline["model"]["backbone"]
    assert posthoc == baseline


def test_concat_passes_norm_kwarg(monkeypatch):
    # No additional checkpoint is needed to check constructor propagation.
    wrn = Mock(return_value=torch.nn.Identity())
    dino = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(backbones, "WRN50Features", wrn)
    monkeypatch.setattr(backbones, "DINOv2Features", dino)
    WRN50DINOv2Concat(norm=False, pretrained=False)
    wrn.assert_called_once_with(layers=("layer2", "layer3"), pretrained=False)
    dino.assert_called_once_with(layers=("block5", "block8", "block11"),
                                 input_size=448, pretrained=False, norm=False)


@pytest.mark.gpu
@pytest.mark.parametrize("norm", [True, False])
def test_dinov2_norm_kwarg_matches_requested_intermediates(norm):
    model = DINOv2Features(layers=("block5", "block8", "block11"), norm=norm).cuda()
    assert model.norm is norm
    images = torch.randn(1, 3, 224, 224, device="cuda")
    actual = model(images)
    with torch.inference_mode():
        resized = F.interpolate(images, (448, 448), mode="bilinear", align_corners=False)
        raw = model.model.forward_intermediates(
            resized, indices=[5, 8, 11], norm=False, output_fmt="NLC", intermediates_only=True)
        for feature, tokens in zip(actual, raw):
            expected = model.model.norm(tokens) if norm else tokens
            torch.testing.assert_close(feature.flatten(2).transpose(1, 2), expected)


@pytest.mark.gpu
def test_raw_build_name_disables_norm_even_if_explicitly_requested():
    model = build_backbone("dinov2_vitb14_raw", norm=True)
    assert model.norm is False
