"""Experiment runner.

Owner: branch feat/pipeline.

CLI:  python -m ad.run --config configs/patchcore_wrn50.yaml [--categories bottle tile]
                       [--max-train 20] [--out results] [--no-figs]
                       [--set key.sub=value ...]

Per category, writes results/<name>/<category>/:
    scores.npz   image_scores (N,), labels (N,), defect_types (N,), paths (N,),
                 val_image_scores (M,)
    maps.npz     maps (N,H,W) float16 post-processed test maps, masks (N,H,W) uint8
    metrics.json ad.metrics.evaluate(...) + image_threshold, pixel_threshold,
                 fit_seconds, infer_ms_per_image, peak_gpu_mem_gb, model_stats, config,
                 category, is_texture
    figs/        overlay panels (ad.visualize.save_category_figures)
Also writes results/<name>/summary.json with categories and their numeric mean.
Inference timing includes test prediction and shared score post-processing, not evaluation.
Thresholds come only from the normal validation split (ad.threshold).
Run GPU experiments under scripts/gpu_run.sh; this module does not acquire the GPU lock.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ad import metrics, scoring, threshold, visualize
from ad.base import Prediction, build_model
from ad.config import deep_merge, load_config
from ad.data import CATEGORIES, TEXTURE_CATEGORIES, MVTecDataset, get_root, make_loader

LOGGER = logging.getLogger(__name__)


def _parse_overrides(assignments: list[str]) -> dict:
    overrides = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator or any(not part for part in key.split(".")):
            raise ValueError(f"invalid --set {assignment!r}; expected key.sub=value")
        nested = yaml.safe_load(value)
        for part in reversed(key.split(".")):
            nested = {part: nested}
        overrides = deep_merge(overrides, nested)
    return overrides


def _write_json(path: Path, value) -> None:
    # Undefined metrics (e.g. AU-PRO without GT regions) use JSON null, not NaN.
    def serializable(item):
        if isinstance(item, dict):
            return {key: serializable(value) for key, value in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [serializable(value) for value in item]
        if isinstance(item, np.generic):
            return serializable(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def _postprocess(pred: Prediction, cfg: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    options = cfg["scoring"]
    maps = scoring.upsample_and_smooth(
        pred.patch_maps, size=cfg["data"]["crop"], sigma=options["sigma"], device=device)
    mode = options["image_score"]
    if mode == "model" and pred.image_scores is not None:
        scores = np.asarray(pred.image_scores, dtype=np.float32)
    else:
        scores = scoring.image_scores_from_maps(
            maps, mode="max" if mode == "model" else mode, topk_ratio=options["topk_ratio"])
    if scores.shape != (len(maps),):
        raise ValueError("model image_scores must have shape (N,)")
    return maps, scores


def _run_category(category: str, cfg: dict, result_dir: Path, no_figs: bool) -> dict:
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    data = cfg["data"]
    dataset_options = {key: data[key] for key in ("root", "resize", "crop", "val_ratio", "seed")}
    datasets = {
        split: MVTecDataset(category, split, **dataset_options,
                            max_items=data["max_train"] if split == "train" else None)
        for split in ("train", "val", "test")
    }
    for split, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f"{category}: {split} split is empty; normal validation is required")
    loaders = {
        split: make_loader(dataset, batch_size=data["batch_size"],
                           num_workers=data["num_workers"], shuffle=False)
        for split, dataset in datasets.items()
    }

    device = torch.device(cfg["model"].get("device", "cuda"))
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model = None
    try:
        model = build_model(cfg["method"], **cfg["model"])
        device = torch.device(model.device)
        use_cuda = device.type == "cuda" and torch.cuda.is_available()
        synchronize()
        started = time.perf_counter()
        model.fit(loaders["train"])
        synchronize()
        fit_seconds = time.perf_counter() - started

        val_maps, val_scores = _postprocess(model.predict(loaders["val"]), cfg, str(device))
        if len(val_maps) != len(datasets["val"]):
            raise ValueError("validation predictions do not match the dataset length")
        image_thr = threshold.image_threshold(val_scores, cfg["threshold"]["image_quantile"])
        pixel_thr = threshold.pixel_threshold(val_maps, cfg["threshold"]["pixel_quantile"])
        del val_maps

        synchronize()
        started = time.perf_counter()
        maps, image_scores = _postprocess(model.predict(loaders["test"]), cfg, str(device))
        synchronize()
        infer_ms = 1000 * (time.perf_counter() - started) / len(datasets["test"])
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if use_cuda else 0.0
        if len(maps) != len(datasets["test"]):
            raise ValueError("test predictions do not match the dataset length")

        # Annotations are collected only after prediction and calibration, for
        # evaluation and output. Sequential loading preserves prediction order.
        labels, masks, defect_types, paths = [], [], [], []
        for batch in loaders["test"]:
            labels.append(batch["label"].numpy())
            masks.append(batch["mask"][:, 0].numpy().astype(np.uint8))
            defect_types.extend(batch["defect_type"])
            paths.extend(batch["path"])
        labels, masks = np.concatenate(labels), np.concatenate(masks)
        result = metrics.evaluate(labels, image_scores, masks, maps, image_thr, pixel_thr)
        result.update({
            "image_threshold": image_thr,
            "pixel_threshold": pixel_thr,
            "fit_seconds": fit_seconds,
            "infer_ms_per_image": infer_ms,
            "peak_gpu_mem_gb": peak_gb,
            "model_stats": model.stats(),
            "category": category,
            "is_texture": category in TEXTURE_CATEGORIES,
            "config": cfg,
        })
        result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_dir / "scores.npz", image_scores=image_scores, labels=labels,
                            defect_types=np.asarray(defect_types), paths=np.asarray(paths),
                            val_image_scores=val_scores)
        np.savez_compressed(result_dir / "maps.npz", maps=maps.astype(np.float16), masks=masks)
        _write_json(result_dir / "metrics.json", result)
        if not no_figs:
            try:
                visualize.save_category_figures(str(result_dir))
            except NotImplementedError:
                LOGGER.warning("Visualization is not implemented; skipping figures for %s", category)
        LOGGER.info("%s: image AUROC=%.4f, pixel AUROC=%.4f, AU-PRO=%.4f",
                    category, result["image_auroc"], result["pixel_auroc"], result["aupro"])
        return result
    finally:
        del model
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-figs", action="store_true")
    parser.add_argument("--set", dest="assignments", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    try:
        overrides = _parse_overrides(args.assignments)
        if args.categories is not None:
            overrides = deep_merge(overrides, {"categories": args.categories})
        if args.max_train is not None:
            overrides = deep_merge(overrides, {"data": {"max_train": args.max_train}})
        cfg = load_config(args.config, overrides)
    except (ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    categories = CATEGORIES if cfg["categories"] is None else cfg["categories"]
    if (not isinstance(categories, (list, tuple)) or not categories
            or any(not isinstance(category, str) or not category for category in categories)):
        parser.error("categories must be a nonempty list of category names")
    cfg["categories"] = list(dict.fromkeys(categories))
    if cfg["data"]["max_train"] is not None and cfg["data"]["max_train"] < 1:
        parser.error("max_train must be positive")
    cfg["data"]["root"] = str(get_root(cfg["data"].get("root")).resolve())
    cfg["model"].setdefault("seed", cfg["seed"])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = args.out / cfg["name"]
    categories = {}
    for category in cfg["categories"]:
        categories[category] = _run_category(category, cfg, run_dir / category, args.no_figs)
    numeric_keys = {key for result in categories.values() for key, value in result.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)}
    mean = {}
    for key in sorted(numeric_keys):
        values = [result[key] for result in categories.values()
                  if key in result and math.isfinite(result[key])]
        mean[key] = float(np.mean(values)) if values else None
    _write_json(run_dir / "summary.json", {"name": cfg["name"], "categories": categories, "mean": mean})


if __name__ == "__main__":
    main()
