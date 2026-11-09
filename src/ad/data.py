"""MVTec AD dataset loading and preprocessing.

Splits
------
- ``train``: 90% of ``<cat>/train/good`` (normal only) -> used to fit models.
- ``val``:   10% of ``<cat>/train/good`` (normal only, fixed seed) -> used ONLY for
  thresholds and score normalisation. No defect labels or masks are ever involved.
- ``test``:  ``<cat>/test/*`` with ground-truth masks -> used ONLY for evaluation.

Preprocessing: resize to ``resize`` x ``resize`` (bilinear for images, nearest for masks),
centre crop to ``crop`` x ``crop``, ImageNet mean/std normalisation. This matches the
PaDiM / PatchCore papers' protocol (256 -> 224) so results are comparable to the literature.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OBJECT_CATEGORIES = ("bottle", "capsule", "screw")
TEXTURE_CATEGORIES = ("carpet", "tile", "leather")
CATEGORIES = OBJECT_CATEGORIES + TEXTURE_CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def get_root(root: str | os.PathLike | None = None) -> Path:
    """MVTec AD root. Priority: explicit arg > $MVTEC_ROOT > <repo>/data/mvtec_ad."""
    if root is not None:
        return Path(root)
    env = os.environ.get("MVTEC_ROOT")
    return Path(env) if env else REPO_ROOT / "data" / "mvtec_ad"


def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def split_train_val(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministic split of normal training images into fit / validation subsets."""
    if val_ratio <= 0:
        return list(paths), []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    n_val = max(1, math.ceil(len(paths) * val_ratio))
    val_idx = set(perm[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


class MVTecDataset(Dataset):
    """One MVTec AD category.

    Each item is a dict:
        image       float32 tensor (3, crop, crop), ImageNet-normalised
        mask        float32 tensor (1, crop, crop), values in {0, 1} (all zeros for normal)
        label       int, 0 = normal, 1 = anomalous
        defect_type str, "good" for normal images
        path        str, absolute image path
    """

    def __init__(
        self,
        category: str,
        split: str,
        root: str | os.PathLike | None = None,
        resize: int = 256,
        crop: int = 224,
        val_ratio: float = 0.1,
        seed: int = 0,
        max_items: int | None = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.category, self.split = category, split
        self.root = get_root(root) / category
        if not self.root.is_dir():
            raise FileNotFoundError(f"category folder not found: {self.root} (set MVTEC_ROOT?)")
        self.resize, self.crop = resize, crop

        # samples: (image_path, mask_path | None, label, defect_type)
        self.samples: list[tuple[Path, Path | None, int, str]] = []
        if split in ("train", "val"):
            fit, val = split_train_val(_list_images(self.root / "train" / "good"), val_ratio, seed)
            for p in fit if split == "train" else val:
                self.samples.append((p, None, 0, "good"))
        else:
            for d in sorted((self.root / "test").iterdir()):
                if not d.is_dir():
                    continue
                for p in _list_images(d):
                    if d.name == "good":
                        self.samples.append((p, None, 0, "good"))
                    else:
                        m = self.root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                        self.samples.append((p, m, 1, d.name))
        if max_items is not None:
            self.samples = self.samples[:max_items]

    def __len__(self) -> int:
        return len(self.samples)

    def _geom(self, img: Image.Image, interp: TF.InterpolationMode) -> Image.Image:
        img = TF.resize(img, [self.resize, self.resize], interpolation=interp)
        return TF.center_crop(img, [self.crop, self.crop])

    def __getitem__(self, idx: int) -> dict:
        path, mask_path, label, dtype = self.samples[idx]
        img = self._geom(Image.open(path).convert("RGB"), TF.InterpolationMode.BILINEAR)
        image = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        if mask_path is not None and mask_path.exists():
            m = self._geom(Image.open(mask_path).convert("L"), TF.InterpolationMode.NEAREST)
            mask = (TF.to_tensor(m) > 0.5).float()
        else:
            mask = torch.zeros(1, self.crop, self.crop)
        return {"image": image, "mask": mask, "label": label, "defect_type": dtype, "path": str(path)}


def make_loader(ds: Dataset, batch_size: int = 32, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def denormalize(image: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) uint8 RGB array for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    x = (image * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
