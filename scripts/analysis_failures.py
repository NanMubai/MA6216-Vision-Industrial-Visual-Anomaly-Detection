"""Failure-case analysis of the PaDiM / PatchCore baseline runs (read-only, CPU only).

Reads the saved per-image outputs written by ``ad.run``
(``results/<run>/<category>/{scores.npz,maps.npz,metrics.json}``) and writes
markdown + CSV tables to ``reports/figures/failure_cases/tables/``.

Nothing here fits a model, changes a threshold or selects a hyper-parameter.
Test labels and masks are read only to *describe* the fixed, already-evaluated
predictions (post-hoc analysis, like ``ad.metrics``). All thresholds are the
validation-derived values stored in ``metrics.json``:

* image threshold = 99% quantile of normal-validation image scores;
* pixel threshold = 99.9% quantile of normal-validation pixel scores.

Tables
------
recall_by_defect_type   image recall per defect type at the val image threshold,
                        plus the false-positive rate on ``good`` test images.
region_detection        per GT region (8-connected component of the 224x224 test
                        mask): detected if >= 30% of its pixels have map >= pixel
                        threshold (PRO-style overlap). Binned by region area
                        measured in WRN50 stride-8 feature cells (1 cell = 8x8 px
                        = 0.128% of a 224x224 crop).
recall_by_image_area    image recall vs total GT area of the image (same bins).
miss_overlap            per defect type: images missed by both methods, only by
                        PaDiM, only by PatchCore (val threshold).
localisation            TPs: peak-outside-GT rate and pixel IoU at the pixel thr.
false_positives         normal test images above the image threshold: where the
                        peak lies (border band vs interior) and its score ratio.
calibration             normal-validation score spread vs the chosen threshold.
per_image.csv           one row per (run, category, test image).

Usage::

    PYTHONPATH=src python scripts/analysis_failures.py \
        [--results /abs/path/results] [--out reports/figures/failure_cases/tables]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

RESULTS = Path("/home/eva/code/ma6216/industrial-visual-anomaly-detection/results")
DATASET_REGIONS = Path("reports/dataset/defect_regions_per_image.csv")  # full-resolution GT areas
RUNS = {"padim_wrn50": "PaDiM", "patchcore_wrn50": "PatchCore"}
CATEGORIES = ("bottle", "capsule", "screw", "carpet", "tile", "leather")
OVERLAP = 0.3          # PRO-style region overlap needed to call a region "detected"
CELL_PX = 8 * 8        # one WRN50 layer2 / stride-8 feature cell in a 224x224 crop
BORDER_PX = 16         # "border band" width for the FP peak location (2 feature cells)
# Region-area bins in feature cells (upper edges, inclusive lower edge).
BINS = [0, 1, 3, 10, 30, 100, np.inf]
BIN_LABELS = ["<1 cell", "1-3 cells", "3-10 cells", "10-30 cells", "30-100 cells", ">=100 cells"]
EIGHT = np.ones((3, 3), dtype=np.uint8)


def _bin(area_px: float) -> str:
    cells = area_px / CELL_PX
    idx = int(np.searchsorted(BINS, cells, side="right")) - 1
    return BIN_LABELS[min(max(idx, 0), len(BIN_LABELS) - 1)]


def load(run: str, category: str, results: Path):
    rdir = results / run / category
    sc = np.load(rdir / "scores.npz", allow_pickle=False)
    mp = np.load(rdir / "maps.npz", allow_pickle=False)
    meta = json.loads((rdir / "metrics.json").read_text())
    return sc, mp, meta


def analyse_run_category(run: str, category: str, results: Path):
    sc, mp, meta = load(run, category, results)
    labels = sc["labels"].astype(int)
    scores = sc["image_scores"].astype(np.float64)
    dtypes = sc["defect_types"].astype(str)
    paths = sc["paths"].astype(str)
    val = sc["val_image_scores"].astype(np.float64)
    maps = mp["maps"]                      # (N,224,224) float16, smoothed
    masks = mp["masks"].astype(bool)
    image_thr = float(meta["image_threshold"])
    pixel_thr = float(meta["pixel_threshold"])
    size = maps.shape[1]

    image_rows, region_rows = [], []
    for i in range(len(labels)):
        amap = maps[i].astype(np.float32)
        gt = masks[i]
        pred = amap >= pixel_thr
        peak = np.unravel_index(int(np.argmax(amap)), amap.shape)
        border = min(peak[0], peak[1], size - 1 - peak[0], size - 1 - peak[1]) < BORDER_PX
        row = {
            "run": run, "method": RUNS[run], "category": category,
            "defect_type": dtypes[i], "file": Path(paths[i]).name, "index": i,
            "label": labels[i], "score": scores[i], "image_thr": image_thr,
            "score_ratio": scores[i] / image_thr, "detected": bool(scores[i] >= image_thr),
            "pixel_thr": pixel_thr, "map_max_ratio": float(amap.max()) / pixel_thr,
            "pred_area_frac": float(pred.mean()),
            "peak_y": int(peak[0]), "peak_x": int(peak[1]), "peak_in_border": bool(border),
        }
        if labels[i] == 1:
            gt_area = int(gt.sum())
            inter = int((pred & gt).sum())
            union = int((pred | gt).sum())
            comps, count = ndimage.label(gt, structure=EIGHT)
            overlaps = []
            for rid, bounds in enumerate(ndimage.find_objects(comps, count), start=1):
                region = comps[bounds] == rid
                area = int(region.sum())
                ov = float(pred[bounds][region].mean())
                overlaps.append(ov)
                region_rows.append({
                    "run": run, "method": RUNS[run], "category": category,
                    "defect_type": dtypes[i], "file": Path(paths[i]).name, "region": rid,
                    "area_px": area, "area_frac": area / gt.size, "area_cells": area / CELL_PX,
                    "size_bin": _bin(area), "overlap": ov, "region_detected": ov >= OVERLAP,
                    "image_detected": bool(scores[i] >= image_thr),
                    "max_score_ratio_in_region": float(amap[bounds][region].max()) / pixel_thr,
                })
            row.update({
                "gt_area_px": gt_area, "gt_area_frac": gt_area / gt.size,
                "gt_size_bin": _bin(gt_area) if gt_area else "no mask in crop",
                "n_regions": int(count),
                "pixel_iou": inter / union if union else np.nan,
                "peak_in_gt": bool(gt[peak]),
                "max_in_gt_ratio": float(amap[gt].max()) / pixel_thr if gt_area else np.nan,
                "max_out_gt_ratio": float(amap[~gt].max()) / pixel_thr,
                "min_region_overlap": min(overlaps) if overlaps else np.nan,
            })
        image_rows.append(row)

    calib = {
        "run": run, "method": RUNS[run], "category": category, "n_val": len(val),
        "val_median": float(np.median(val)), "val_max": float(val.max()),
        "val_2nd_max": float(np.sort(val)[-2]), "image_thr": image_thr,
        "thr_over_val_median": image_thr / float(np.median(val)),
        "val_cv": float(val.std() / val.mean()),
        "test_good_max": float(scores[labels == 0].max()),
        "test_good_median": float(np.median(scores[labels == 0])),
        "test_anom_median": float(np.median(scores[labels == 1])),
        "fpr_good": float((scores[labels == 0] >= image_thr).mean()),
        "recall": float((scores[labels == 1] >= image_thr).mean()),
        "precision": float(meta["image_precision"]),
        "image_f1": float(meta["image_f1"]),
        "image_auroc": float(meta["image_auroc"]),
    }
    return image_rows, region_rows, calib


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal GitHub markdown table (the shared venv has no ``tabulate``)."""
    def cell(v) -> str:
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    numeric = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---:" if n else "---" for n in numeric) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=Path("reports/figures/failure_cases/tables"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    images, regions, calib = [], [], []
    for run in RUNS:
        for cat in CATEGORIES:
            a, b, c = analyse_run_category(run, cat, args.results)
            images += a
            regions += b
            calib.append(c)
    img = pd.DataFrame(images)
    reg = pd.DataFrame(regions)
    cal = pd.DataFrame(calib)
    # Full-resolution GT area from the dataset analysis -> share of each defect kept by the
    # resize-256 / centre-crop-224 preprocessing (crop covers (224/256)^2 of the image).
    full = pd.read_csv(DATASET_REGIONS)[["category", "defect_type", "file", "area_frac"]]
    img = img.merge(full.rename(columns={"area_frac": "gt_area_frac_full"}), how="left",
                    on=["category", "defect_type", "file"])
    crop_share = (224 / 256) ** 2
    img["gt_retained_in_crop"] = np.clip(img.gt_area_frac * crop_share / img.gt_area_frac_full, 0, 1)
    img.to_csv(args.out / "per_image.csv", index=False, float_format="%.6g")
    reg.to_csv(args.out / "per_region.csv", index=False, float_format="%.6g")

    sections = []

    # 1. Recall per defect type at the validation image threshold.
    anom = img[img.label == 1]
    rec = (anom.groupby(["category", "defect_type", "method"])
           .agg(n=("detected", "size"), detected=("detected", "sum"))
           .reset_index())
    rec["recall"] = rec.detected / rec.n
    wide = rec.pivot_table(index=["category", "defect_type", "n"], columns="method",
                           values="recall").reset_index()
    good = img[img.label == 0]
    fp = (good.groupby(["category", "method"]).agg(n=("detected", "size"), fp=("detected", "sum"))
          .reset_index())
    fp["fpr"] = fp.fp / fp.n
    fpw = fp.pivot_table(index=["category", "n"], columns="method", values="fpr").reset_index()
    fpw.insert(1, "defect_type", "good (FPR)")
    order = {c: k for k, c in enumerate(CATEGORIES)}
    tab = pd.concat([wide, fpw], ignore_index=True)
    tab = tab.sort_values(["category", "defect_type"], key=lambda s: s.map(order) if s.name == "category"
                          else s.str.startswith("good").astype(int).astype(str) + s)
    tab = tab[["category", "defect_type", "n", "PaDiM", "PatchCore"]]
    tab.to_csv(args.out / "recall_by_defect_type.csv", index=False, float_format="%.4f")
    sections.append(("recall_by_defect_type",
                     "Image-level recall per defect type at the normal-validation 99% image "
                     "threshold. Rows `good (FPR)` give the false-positive rate on normal test "
                     "images at the same threshold.", md(tab)))

    # 2. Region detection vs region size (PRO-style overlap >= 0.3 at the pixel threshold).
    def region_table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        g = (frame.groupby(by + ["size_bin", "method"], observed=True)
             .agg(n=("region_detected", "size"), rate=("region_detected", "mean")).reset_index())
        w = g.pivot_table(index=by + ["size_bin", "n"], columns="method", values="rate").reset_index()
        w["size_bin"] = pd.Categorical(w.size_bin, BIN_LABELS, ordered=True)
        return w.sort_values(by + ["size_bin"])

    reg["size_bin"] = pd.Categorical(reg.size_bin, BIN_LABELS, ordered=True)
    pooled = region_table(reg, [])
    cat_cells = reg.assign(category=pd.Categorical(reg.category, CATEGORIES, ordered=True))
    percat = region_table(cat_cells, ["category"])
    pooled.to_csv(args.out / "region_detection_pooled.csv", index=False, float_format="%.4f")
    percat.to_csv(args.out / "region_detection_by_category.csv", index=False, float_format="%.4f")
    reg_summary = (reg.groupby(["category", "method"]).region_detected.mean()
                   .unstack().reindex(CATEGORIES).reset_index())
    sections.append(("region_detection_pooled",
                     f"Share of GT regions (8-connected, 224x224 crop) with >= {OVERLAP:.0%} of their "
                     "pixels above the validation pixel threshold, pooled over all six categories, "
                     f"by region area in stride-8 feature cells (1 cell = {CELL_PX} px = "
                     f"{CELL_PX / 224 ** 2:.3%} of the crop).", md(pooled)))
    sections.append(("region_detection_by_category", "Same, per category.", md(percat)))
    sections.append(("region_detection_overall", "Region detection rate per category (all sizes).",
                     md(reg_summary)))

    # 3. Image recall vs total GT area.
    anom = anom.assign(gt_size_bin=pd.Categorical(anom.gt_size_bin, BIN_LABELS, ordered=True))
    ir = (anom.groupby(["gt_size_bin", "method"], observed=True)
          .agg(n=("detected", "size"), recall=("detected", "mean")).reset_index()
          .pivot_table(index=["gt_size_bin", "n"], columns="method", values="recall",
                       observed=True).reset_index())
    ir.to_csv(args.out / "recall_by_image_area.csv", index=False, float_format="%.4f")
    sections.append(("recall_by_image_area",
                     "Image-level recall (val threshold) vs total GT area of the image, pooled over "
                     "the six categories.", md(ir)))

    # 4. Missed by both / only one method.
    w = anom.pivot_table(index=["category", "defect_type", "file"], columns="method",
                         values="detected").reset_index()
    w["both_missed"] = (w.PaDiM == 0) & (w.PatchCore == 0)
    w["padim_only_missed"] = (w.PaDiM == 0) & (w.PatchCore == 1)
    w["patchcore_only_missed"] = (w.PaDiM == 1) & (w.PatchCore == 0)
    mo = (w.groupby(["category", "defect_type"])
          .agg(n=("file", "size"), both_missed=("both_missed", "sum"),
               padim_only=("padim_only_missed", "sum"),
               patchcore_only=("patchcore_only_missed", "sum")).reset_index())
    mo = mo.sort_values(["category", "defect_type"], key=lambda s: s.map(order) if s.name == "category" else s)
    mo.to_csv(args.out / "miss_overlap.csv", index=False)
    sections.append(("miss_overlap",
                     "Anomalous test images missed at the val image threshold: by both methods, "
                     "only by PaDiM, only by PatchCore.", md(mo, ".0f")))

    # 4b. Defects cut by the centre crop.
    pc = img[(img.label == 1) & (img.method == "PatchCore")]
    crop = (pc.assign(lt50=pc.gt_retained_in_crop < 0.5, zero=pc.gt_area_px == 0)
            .groupby("category").agg(n_anom=("file", "size"), retained_lt_50pct=("lt50", "sum"),
                                     no_gt_in_crop=("zero", "sum"),
                                     median_retained=("gt_retained_in_crop", "median"))
            .reindex(CATEGORIES).reset_index())
    cut = img[(img.label == 1) & (img.gt_retained_in_crop < 0.5)]
    cutd = (cut.groupby("method").agg(n=("detected", "size"), recall=("detected", "mean")).reset_index())
    crop.to_csv(args.out / "crop_truncation.csv", index=False, float_format="%.4f")
    sections.append(("crop_truncation",
                     "Anomalous test images whose GT is partly removed by resize-256 / centre-crop-224 "
                     "(retained share = GT area inside the crop / full-resolution GT area from "
                     "`reports/dataset/defect_regions_per_image.csv`; approximate because of resize "
                     "interpolation).", md(crop)))
    sections.append(("crop_truncation_recall",
                     "Image recall (val threshold) on images with < 50% of the GT kept in the crop.",
                     md(cutd)))

    # 4c. Recall vs total GT area inside the hard object categories.
    hard = anom[anom.category.isin(["capsule", "screw"])]
    hr = (hard.groupby(["category", "gt_size_bin", "method"], observed=True)
          .agg(n=("detected", "size"), recall=("detected", "mean")).reset_index()
          .pivot_table(index=["category", "gt_size_bin", "n"], columns="method", values="recall",
                       observed=True).reset_index())
    hr.to_csv(args.out / "recall_by_image_area_capsule_screw.csv", index=False, float_format="%.4f")
    sections.append(("recall_by_image_area_capsule_screw",
                     "Image recall vs total GT area, capsule and screw only.", md(hr)))

    # 5. Localisation quality of true positives.
    tp = anom[anom.detected]
    loc = (tp.groupby(["category", "method"])
           .agg(n_tp=("detected", "size"), peak_outside_gt=("peak_in_gt", lambda s: 1 - s.mean()),
                median_pixel_iou=("pixel_iou", "median"),
                iou_below_0_1=("pixel_iou", lambda s: (s.fillna(0) < 0.1).mean()))
           .reset_index())
    loc["category"] = pd.Categorical(loc.category, CATEGORIES, ordered=True)
    loc = loc.sort_values(["category", "method"])
    loc.to_csv(args.out / "localisation.csv", index=False, float_format="%.4f")
    sections.append(("localisation",
                     "True positives (val image threshold): fraction whose map maximum lies outside "
                     "the GT mask, median pixel IoU at the val pixel threshold, and fraction with "
                     "IoU < 0.1.", md(loc)))

    # 6. False positives on normal images.
    fpi = good[good.detected]
    fpt = (good.groupby(["category", "method"])
           .agg(n_good=("detected", "size"), n_fp=("detected", "sum")).reset_index())
    extra = (fpi.groupby(["category", "method"])
             .agg(fp_peak_in_border=("peak_in_border", "mean"),
                  fp_median_score_ratio=("score_ratio", "median"),
                  fp_median_pred_area=("pred_area_frac", "median")).reset_index())
    fpt = fpt.merge(extra, how="left", on=["category", "method"])
    border_share = 1 - ((224 - 2 * BORDER_PX) / 224) ** 2
    fpt["category"] = pd.Categorical(fpt.category, CATEGORIES, ordered=True)
    fpt = fpt.sort_values(["category", "method"])
    fpt.to_csv(args.out / "false_positives.csv", index=False, float_format="%.4f")
    sections.append(("false_positives",
                     f"Normal test images scored above the val image threshold. `fp_peak_in_border`: "
                     f"share of FPs whose map maximum lies within {BORDER_PX} px of the crop edge "
                     f"(this band is {border_share:.0%} of the image area). `fp_median_score_ratio`: "
                     "median image score / image threshold. `fp_median_pred_area`: median fraction "
                     "of pixels above the pixel threshold.", md(fpt)))

    # 7. Calibration.
    cal["category"] = pd.Categorical(cal.category, CATEGORIES, ordered=True)
    cal = cal.sort_values(["method", "category"])
    cal.to_csv(args.out / "calibration.csv", index=False, float_format="%.4f")
    cols = ["method", "category", "n_val", "val_median", "val_2nd_max", "val_max", "image_thr",
            "thr_over_val_median", "val_cv", "test_good_median", "test_good_max", "test_anom_median",
            "fpr_good", "precision", "recall"]
    sections.append(("calibration",
                     "Normal-validation image-score spread vs the 99% quantile threshold. With "
                     "n_val = 21-32 the 99% quantile interpolates between the two largest val "
                     "scores. `val_cv` = std/mean of val scores.", md(cal[cols])))

    text = ["# Failure-case analysis tables",
            "",
            "Generated by `scripts/analysis_failures.py` from the saved outputs of the "
            "`padim_wrn50` and `patchcore_wrn50` runs (results/<run>/<category>/). Thresholds are "
            "the stored normal-validation values; test labels and masks are used only to describe "
            "the fixed predictions. Maps and masks are at the 224x224 evaluation crop, so region "
            "areas are fractions of the crop (the dataset analysis reports areas at the original "
            "full resolution).",
            ""]
    for name, caption, table in sections:
        text += [f"## {name}", "", caption, "", table, ""]
    (args.out / "failure_tables.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
