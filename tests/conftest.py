import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def fake_mvtec(tmp_path):
    """Tiny MVTec-style tree: one category 'toy' with 20 train/good, 3 test/good, 2 test/crack."""
    root = tmp_path / "mvtec"
    rng = np.random.default_rng(0)

    def img(p, size=(300, 300)):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(p)

    for i in range(20):
        img(root / "toy" / "train" / "good" / f"{i:03d}.png")
    for i in range(3):
        img(root / "toy" / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        img(root / "toy" / "test" / "crack" / f"{i:03d}.png")
        m = np.zeros((300, 300), np.uint8)
        m[100:200, 120:180] = 255
        p = root / "toy" / "ground_truth" / "crack" / f"{i:03d}_mask.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(m).save(p)
    return root


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA")
        for it in items:
            if "gpu" in it.keywords:
                it.add_marker(skip)
