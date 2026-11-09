"""YAML experiment configs merged over DEFAULTS.

Schema (all keys optional in a YAML file except ``name`` and ``method``):

    name: patchcore_wrn50          # run name -> results/<name>/
    method: patchcore              # "padim" | "patchcore"
    categories: [bottle, ...]      # default: ad.data.CATEGORIES
    data:      {resize, crop, val_ratio, seed, batch_size, num_workers, max_train}
    model:     {...}               # passed as **kwargs to the model constructor
    scoring:   {sigma, image_score, topk_ratio}
    threshold: {image_quantile, pixel_quantile}
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": None,
    "method": None,
    "categories": None,
    "seed": 0,
    "data": {
        "resize": 256,
        "crop": 224,
        "val_ratio": 0.1,
        "seed": 0,
        "batch_size": 32,
        "num_workers": 4,
        "max_train": None,  # cap on #fit images, for smoke tests only
    },
    "model": {},
    "scoring": {
        "sigma": 4.0,               # Gaussian smoothing of upsampled anomaly map (pixels)
        "image_score": "model",     # "model" (model-provided if any, else max) | "max" | "topk_mean"
        "topk_ratio": 0.01,         # fraction of pixels for topk_mean
    },
    "threshold": {
        "image_quantile": 0.99,     # on normal validation image scores
        "pixel_quantile": 0.999,    # on normal validation pixel scores
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = deep_merge(DEFAULTS, yaml.safe_load(f) or {})
    cfg = deep_merge(cfg, overrides or {})
    if not cfg["name"] or not cfg["method"]:
        raise ValueError(f"{path}: config needs 'name' and 'method'")
    return cfg
