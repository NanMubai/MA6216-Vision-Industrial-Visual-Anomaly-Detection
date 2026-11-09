"""Experiment runner.

Owner: branch feat/pipeline.

CLI:  python -m ad.run --config configs/patchcore_wrn50.yaml [--categories bottle tile]
                       [--max-train 20] [--out results] [--no-figs]
                       [--set key.sub=value ...]

Per category, writes results/<name>/<category>/:
    scores.npz   image_scores (N,), labels (N,), defect_types (N,), paths (N,),
                 val_image_scores (M,)
    maps.npz     maps (N,H,W) float16 post-processed test maps (float32 with zscore), masks (N,H,W) uint8
    metrics.json ad.metrics.evaluate(...) + image_threshold, pixel_threshold,
                 fit_seconds, infer_ms_per_image, infer_ms_per_image_e2e,
                 peak_gpu_mem_gb, model_stats, config, category, is_texture
    error.json   category, error_type, error, traceback, config (on category failure)
    figs/        overlay panels (ad.visualize.save_category_figures)
Also writes results/<name>/summary.json with successful categories, errors, and numeric
means excluding thresholds. A category failure is logged and later categories continue;
the command exits nonzero after writing the summary if any category failed.
infer_ms_per_image times only model.predict plus upsampling/smoothing on preloaded
test batches, with CUDA synchronisation. infer_ms_per_image_e2e also includes loading
those batches (including worker startup). Both exclude evaluation and image-score
aggregation. The preloaded CPU batches are released after their masks are collected.
Thresholds come only from the normal validation split (ad.threshold).
scoring.normalize defaults to none; zscore fits separate image/pixel statistics
on normal validation and reports image_f1_global (fixed z=3) and
image_f1_global_pooled (99% pooled normalised validation image quantile) after
all requested categories succeed. Per-category image_f1 remains available.
--prediction-cache optionally shares raw model outputs across postprocessing
variants with identical model/data/seed configuration. Recorded inference time
adds the cached model prediction time to each newly measured smoothing pass;
fit time and model peak memory are inherited and marked in metrics.json.
Run GPU experiments under scripts/gpu_run.sh; this module does not acquire the GPU lock.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
import traceback
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


class _PreloadedBatches(list):
    """A reusable batch iterable retaining the dataset metadata expected by loaders."""

    def __init__(self, loader):
        super().__init__(loader)
        self.dataset = loader.dataset


def _image_scores(pred: Prediction, maps: np.ndarray, cfg: dict, mode: str) -> np.ndarray:
    options = cfg["scoring"]
    if mode == "model":
        scores = np.asarray(pred.image_scores, dtype=np.float32)
    else:
        scores = scoring.image_scores_from_maps(
            maps, mode=mode, topk_ratio=options["topk_ratio"])
    if scores.shape != (len(maps),):
        raise ValueError("model image_scores must have shape (N,)")
    return scores


def _save_raw_prediction(path: Path, prediction: Prediction) -> None:
    arrays = {"patch_maps": prediction.patch_maps}
    if prediction.image_scores is not None:
        arrays["image_scores"] = prediction.image_scores
    np.savez_compressed(path, **arrays)


class _CachedPredictions:
    """Replay raw model outputs to compare postprocessing without refitting."""

    def __init__(self, path: Path, metadata: dict, device):
        self.path, self.metadata, self.device = path, metadata, device

    def fit(self, loader):
        pass

    def predict(self, loader):
        with np.load(self.path / f"{loader.dataset.split}.npz", allow_pickle=False) as arrays:
            return Prediction(arrays["patch_maps"], arrays.get("image_scores"))

    def stats(self):
        return self.metadata["model_stats"]


def _apply_global_thresholds(run_dir: Path, categories: dict) -> None:
    """Calibrate from validation, then delegate all test-label evaluation to metrics."""
    val_scores = {}
    for category in categories:
        with np.load(run_dir / category / "scores.npz", allow_pickle=False) as arrays:
            val_scores[category] = arrays["val_image_scores"]
    thresholds = {"global": threshold.global_image_threshold(val_scores),
                  "global_pooled": threshold.global_image_threshold(val_scores, "pooled_quantile")}
    for category, result in categories.items():
        with np.load(run_dir / category / "scores.npz", allow_pickle=False) as arrays:
            for suffix, value in thresholds.items():
                measured = metrics.f1_at_threshold(arrays["labels"], arrays["image_scores"], value)
                result[f"image_threshold_{suffix}"] = value
                result.update({f"image_{key}_{suffix}": number for key, number in measured.items()})
        _write_json(run_dir / category / "metrics.json", result)
    _write_json(run_dir / "global_thresholds.json", {
        "source": "normal_validation", "categories": list(categories),
        "image_threshold_global": thresholds["global"],
        "image_threshold_global_pooled": thresholds["global_pooled"], "pooled_quantile": 0.99,
        "val_image_count": {key: len(value) for key, value in val_scores.items()}})


def _run_category(category: str, cfg: dict, result_dir: Path, no_figs: bool,
                  prediction_cache: Path | None = None) -> dict:
    normalize = cfg["scoring"].get("normalize", "none")
    if normalize not in ("none", "zscore"):
        raise ValueError("scoring.normalize must be none or zscore")
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
        # A cache is opt-in and reusable ONLY when every model/data setting matches.
        # It holds raw predictions, never labels/masks or calibrated statistics.
        cache_path = prediction_cache / category if prediction_cache is not None else None
        identity = {key: value for key, value in cfg.items()
                    if key not in ("name", "scoring", "threshold")}
        cached = None
        if cache_path is not None and (cache_path / "metadata.json").is_file():
            cached = json.loads((cache_path / "metadata.json").read_text())
            if cached["identity"] != identity:
                raise ValueError("prediction cache model/data config differs from this run")
        if cached is None:
            model = build_model(cfg["method"], **cfg["model"])
        else:
            model = _CachedPredictions(cache_path, cached, device)
        device = torch.device(model.device)
        use_cuda = device.type == "cuda" and torch.cuda.is_available()
        synchronize()
        started = time.perf_counter()
        model.fit(loaders["train"])
        synchronize()
        fit_seconds = (cached["fit_seconds"] if cached is not None
                       else time.perf_counter() - started)

        val_pred = model.predict(loaders["val"])
        model_scores_available = val_pred.image_scores is not None
        score_mode = cfg["scoring"]["image_score"]
        if score_mode == "model" and not model_scores_available:
            score_mode = "max"
        val_maps = scoring.upsample_and_smooth(
            val_pred.patch_maps, size=data["crop"], sigma=cfg["scoring"]["sigma"], device=str(device))
        val_scores = _image_scores(val_pred, val_maps, cfg, score_mode)
        if len(val_maps) != len(datasets["val"]):
            raise ValueError("validation predictions do not match the dataset length")
        normalization = None
        if normalize == "zscore":
            normalization = {"source": "normal_validation", "image": scoring.fit_zscore(val_scores),
                             "pixel": scoring.fit_zscore(val_maps)}
            val_scores = scoring.apply_zscore(val_scores, normalization["image"])
            val_maps = scoring.apply_zscore(val_maps, normalization["pixel"])
        image_thr = threshold.image_threshold(val_scores, cfg["threshold"]["image_quantile"])
        pixel_thr = threshold.pixel_threshold(val_maps, cfg["threshold"]["pixel_quantile"])
        if cache_path is not None and cached is None:
            cache_path.mkdir(parents=True, exist_ok=True)
            _save_raw_prediction(cache_path / "val.npz", val_pred)
        del val_maps, val_pred

        synchronize()
        e2e_started = time.perf_counter()
        test_batches = _PreloadedBatches(loaders["test"])
        synchronize()
        started = time.perf_counter()
        test_pred = model.predict(test_batches)
        synchronize()
        predicted = time.perf_counter()
        maps = scoring.upsample_and_smooth(
            test_pred.patch_maps, size=data["crop"], sigma=cfg["scoring"]["sigma"], device=str(device))
        synchronize()
        stopped = time.perf_counter()
        predict_seconds = (cached["predict_seconds"] if cached is not None else predicted - started)
        infer_ms = 1000 * (predict_seconds + stopped - predicted) / len(datasets["test"])
        infer_ms_e2e = infer_ms + 1000 * (started - e2e_started) / len(datasets["test"])
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if use_cuda else 0.0
        if cached is not None:
            peak_gb = max(peak_gb, cached["peak_gpu_mem_gb"])
        if cache_path is not None and cached is None:
            _save_raw_prediction(cache_path / "test.npz", test_pred)
            _write_json(cache_path / "metadata.json", {
                "identity": identity, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "peak_gpu_mem_gb": peak_gb, "model_stats": model.stats()})
        if (test_pred.image_scores is not None) != model_scores_available:
            raise ValueError("model image_scores availability differs between validation and test")
        image_scores = _image_scores(test_pred, maps, cfg, score_mode)
        if normalization is not None:
            image_scores = scoring.apply_zscore(image_scores, normalization["image"])
            maps = scoring.apply_zscore(maps, normalization["pixel"])
        if len(maps) != len(datasets["test"]):
            raise ValueError("test predictions do not match the dataset length")

        # Metadata comes directly from the sequential dataset samples. Reuse
        # cached masks after prediction, without another image-loading pass.
        samples = datasets["test"].samples
        labels = np.asarray([sample[2] for sample in samples], dtype=np.uint8)
        defect_types = np.asarray([sample[3] for sample in samples])
        paths = np.asarray([str(sample[0]) for sample in samples])
        masks = np.concatenate([batch["mask"][:, 0].numpy().astype(np.uint8) for batch in test_batches])
        del test_batches, test_pred
        result = metrics.evaluate(labels, image_scores, masks, maps, image_thr, pixel_thr)
        result.update({
            "image_threshold": image_thr,
            "pixel_threshold": pixel_thr,
            "fit_seconds": fit_seconds,
            "infer_ms_per_image": infer_ms,
            "infer_ms_per_image_e2e": infer_ms_e2e,
            "peak_gpu_mem_gb": peak_gb,
            "model_stats": model.stats(),
            "category": category,
            "is_texture": category in TEXTURE_CATEGORIES,
            "config": cfg,
        })
        if normalization is not None:
            result["normalization"] = normalization
            # The same deployment cutoff is also available for normalised maps.
            # These pixel metrics include all test images; existing pixel_f1
            # retains its original anomalous-image-only evaluation convention.
            result["pixel_threshold_global"] = 3.0
            result.update({f"pixel_{key}_global": value for key, value in
                           metrics.f1_at_threshold(masks, maps, 3.0).items()})
        if cache_path is not None:
            result["prediction_cache"] = {"path": str(cache_path), "reused": cached is not None,
                                          "predict_seconds": predict_seconds,
                                          "timing": "shared model prediction plus measured postprocessing"}
        result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_dir / "scores.npz", image_scores=image_scores, labels=labels,
                            defect_types=np.asarray(defect_types), paths=np.asarray(paths),
                            val_image_scores=val_scores)
        # Keep normalised maps in float32: small validation std can produce z
        # scores beyond float16 range, and quantisation can change rank metrics.
        map_dtype = np.float32 if normalization is not None else np.float16
        np.savez_compressed(result_dir / "maps.npz", maps=maps.astype(map_dtype), masks=masks)
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
    parser.add_argument("--prediction-cache", type=Path,
                        help="opt-in raw prediction cache shared by postprocessing variants; "
                             "timing reuses model prediction cost and measures each smoothing pass")
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
    categories, errors = {}, {}
    for category in cfg["categories"]:
        result_dir = run_dir / category
        try:
            cache_options = {"prediction_cache": args.prediction_cache} if args.prediction_cache else {}
            categories[category] = _run_category(category, cfg, result_dir, args.no_figs, **cache_options)
            (result_dir / "error.json").unlink(missing_ok=True)
        except Exception as error:
            LOGGER.exception("Category %s failed", category)
            errors[category] = {"category": category, "error_type": type(error).__name__,
                                "error": str(error), "traceback": traceback.format_exc(), "config": cfg}
            result_dir.mkdir(parents=True, exist_ok=True)
            # A failed rerun must not expose old successful metrics to aggregation.
            (result_dir / "metrics.json").unlink(missing_ok=True)
            _write_json(result_dir / "error.json", errors[category])
    if cfg["scoring"].get("normalize", "none") == "zscore" and not errors:
        _apply_global_thresholds(run_dir, categories)
    else:
        (run_dir / "global_thresholds.json").unlink(missing_ok=True)
    numeric_keys = {key for result in categories.values() for key, value in result.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    and "threshold" not in key}
    mean = {}
    for key in sorted(numeric_keys):
        values = [result[key] for result in categories.values()
                  if isinstance(result.get(key), (int, float)) and math.isfinite(result[key])]
        mean[key] = float(np.mean(values)) if values else None
    _write_json(run_dir / "summary.json", {"name": cfg["name"], "categories": categories,
                                           "errors": errors, "mean": mean})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
