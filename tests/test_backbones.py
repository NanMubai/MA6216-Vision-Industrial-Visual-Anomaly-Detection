import pytest
import torch

from ad.backbones import WRN50Features


@pytest.mark.gpu
def test_wrn50_shapes_and_order():
    m = WRN50Features(layers=("layer3", "layer1")).cuda()
    out = m(torch.randn(2, 3, 224, 224, device="cuda"))
    assert [tuple(o.shape) for o in out] == [(2, 1024, 14, 14), (2, 256, 56, 56)]
    assert m.channels == [1024, 256]
