#!/usr/bin/env python3
"""Render paper tables from committed reports, without running any experiments.

Sources are read with ``git show HEAD:path``: local, uncommitted report edits
cannot silently change a paper. Means are unweighted category means; labels in
the existing evaluation CSVs are used only to summarise completed evaluations.
Run from any directory; --output-dir defaults to paper/generated.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "capsule", "screw", "carpet", "tile", "leather")
GROUPS = {**{c.title(): (c,) for c in CATEGORIES},
          "Obj": CATEGORIES[:3], "Tex": CATEGORIES[3:], "All": CATEGORIES}
RUNS = {"pd": "padim_wrn50", "pc": "patchcore_wrn50"}
METRICS = {
    "ImgAuroc": "image_auroc", "ImgFone": "image_f1",
    "ImgFmax": "image_f1_max_oracle", "PixAuroc": "pixel_auroc",
    "Aupro": "aupro", "Fit": "fit_seconds", "Infer": "infer_ms_per_image",
    "InferEtoE": "infer_ms_per_image_e2e", "Mem": "peak_gpu_mem_gb",
}
BONUS_METRICS = ("ImgAuroc", "ImgFone", "PixAuroc", "Aupro", "Infer", "Mem")
FAILURES = "reports/figures/failure_cases/tables/"


def source_bytes(path: str) -> bytes:
    if not path.startswith(("reports/", "docs/")) or ".." in Path(path).parts:
        raise ValueError(f"not an allowed report source: {path}")
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def source_text(path: str) -> str:
    return source_bytes(path).decode("utf-8")


def csv_rows(path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(source_text(path))))


def md_tables(path: str) -> list[list[dict[str, str]]]:
    """Parse pipe tables, preserving the source's displayed rounding."""
    tables, block = [], []
    for line in [*source_text(path).splitlines(), ""]:
        if line.startswith("|"):
            block.append([x.strip() for x in line.strip().strip("|").split("|")])
        elif block:
            tables.append([dict(zip(block[0], row)) for row in block[2:]])
            block = []
    return tables


class Metrics:
    def __init__(self, path: str):
        self.values = {(r["run"], r["category"], r["metric"]): float(r["value"])
                       for r in csv_rows(path) if r["value"]}

    def get(self, run: str, metric: str, cats=CATEGORIES) -> float:
        return mean(self.values[run, cat, metric] for cat in cats)


@dataclass(frozen=True)
class Variant:
    macro: str
    label: str
    family: str
    run: str
    posthoc: bool = False
    f1_metric: str = "image_f1"


# Order follows report_notes §7, with combined sigma/global rows expanded.
VARIANTS = (
    Variant("Base", "WRN50 baseline (Eq.~7)", "baseline", "patchcore_wrn50"),
    Variant("DinoB", "DINOv2 ViT-B/14", "dinov2", "patchcore_dinov2_vitb14"),
    Variant("DinoS", "DINOv2 ViT-S/14", "dinov2", "patchcore_dinov2_vits14"),
    Variant("WrnDino", "WRN50 + DINOv2-B", "dinov2", "patchcore_wrn50_dinov2"),
    Variant("DinoNoLN", r"DINOv2-B no-LN\posthoc", "dinov2",
            "patchcore_dinov2_vitb14_noln — post-hoc, test-informed (added after a screw diagnostic)", True),
    Variant("MsLOneTwoThree", "Layers 1+2+3", "multiscale", "patchcore_ms_l123"),
    Variant("MsLTwoThreeFour", "Layers 2+3+4", "multiscale", "patchcore_ms_l234"),
    Variant("MsMaxCtrl", "Max-score control", "multiscale", "patchcore_ms_baseline_max"),
    Variant("PyrMean", "Pyramid mean", "multiscale", "patchcore_ms_res_1_1.5_mean"),
    Variant("PyrZmean", "$z$-pyramid mean", "multiscale", "patchcore_ms_res_1_1.5_zmean"),
    Variant("PpTopk", r"Top-1\% mean", "postproc", "pp_topk1"),
    Variant("PpMax", "Smoothed max", "postproc", "pp_max"),
    Variant("PpSigZero", r"Smoothing $\sigma=0$", "postproc", "pp_nosmooth"),
    Variant("PpSigTwo", r"Smoothing $\sigma=2$", "postproc", "pp_sigma2"),
    Variant("PpSigEight", r"Smoothing $\sigma=8$", "postproc", "pp_sigma8"),
    Variant("PpGlobalZ", "Global threshold $z=3$", "postproc", "pp_zscore_global",
            f1_metric="image_f1_global"),
    Variant("PpGlobalPooled", "Global pooled val", "postproc", "pp_zscore_global",
            f1_metric="image_f1_global_pooled"),
)


def load_metrics() -> dict[str, Metrics]:
    return {"baseline": Metrics("reports/tables/metrics_long.csv"),
            **{f: Metrics(f"reports/bonus/{f}_tables/metrics_long.csv")
               for f in ("dinov2", "multiscale", "postproc")}}


def variant_value(v: Variant, metric: str, data: dict[str, Metrics], cats=CATEGORIES) -> float:
    key = v.f1_metric if metric == "ImgFone" else METRICS[metric]
    return data[v.family].get(v.run, key, cats)


def tex_escape(value: str) -> str:
    return re.sub(r"([_&#%])", r"\\\1", value)


def bold(value: str, is_best: bool) -> str:
    return r"\textbf{" + value + "}" if is_best else value


def pair(a: float, b: float, dp=3, lower=False) -> list[str]:
    best = min(a, b) if lower else max(a, b)
    # CSV diagnostics are already rounded to four places. Decimal prevents a
    # stored tie (e.g. 0.5375) from moving due to its binary float representation.
    return [bold(f"{Decimal(str(v)):.{dp}f}", abs(v - best) < 1e-12) for v in (a, b)]


def tabular(spec: str, headers: list[str], rows: list[list[str] | None]) -> str:
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", *headers, r"\midrule"]
    lines.extend(r"\midrule" if row is None else " & ".join(row) + r" \\" for row in rows)
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def failure_counts() -> dict[str, int]:
    rows = csv_rows(FAILURES + "per_image.csv")
    counts = {}
    for prefix, method in (("pd", "PaDiM"), ("pc", "PatchCore")):
        selected = [r for r in rows if r["method"] == method]
        counts[prefix + "Detected"] = sum(r["label"] == "1" and r["detected"] == "True" for r in selected)
        counts[prefix + "FP"] = sum(r["label"] == "0" and r["detected"] == "True" for r in selected)
        n_anom = sum(r["label"] == "1" for r in selected)
        n_good = sum(r["label"] == "0" for r in selected)
        assert (n_anom, n_good) == (556, 177)
    counts.update(nAnomTest=n_anom, nGoodTest=n_good)
    overlap = csv_rows(FAILURES + "miss_overlap.csv")
    for macro, key in (("missBoth", "both_missed"), ("missPdOnly", "padim_only"), ("missPcOnly", "patchcore_only")):
        counts[macro] = sum(int(r[key]) for r in overlap)
    expected = dict(nAnomTest=556, nGoodTest=177, pdDetected=343, pcDetected=484,
                    pdFP=8, pcFP=19, missBoth=61, missPdOnly=152, missPcOnly=11)
    assert counts == expected, (counts, expected)
    return counts


def generate(output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    main = data["baseline"]
    macros = {}
    for prefix, run in RUNS.items():
        for metric, key in METRICS.items():
            for group, cats in GROUPS.items():
                dp = 2 if metric in ("Fit", "Infer", "InferEtoE", "Mem") else 4
                macros[prefix + metric + group] = f"{main.get(run, key, cats):.{dp}f}"
    macros.update({k: str(v) for k, v in failure_counts().items()})
    for v in VARIANTS:
        for metric in BONUS_METRICS:
            dp = 2 if metric in ("Infer", "Mem") else 4
            macros["bn" + v.macro + metric] = f"{variant_value(v, metric, data):.{dp}f}"
    # Cross-check the global F1 values against the published Markdown table.
    global_row = next(r for r in md_tables("reports/bonus/postproc_tables/global_thresholds.md")[0]
                      if r["category"] == "mean (all)")
    assert macros["bnPpGlobalZImgFone"] == global_row["image_f1_global"]
    assert macros["bnPpGlobalPooledImgFone"] == global_row["image_f1_global_pooled"]
    (output / "numbers.tex").write_text(
        "% Generated by scripts/paper_tables.py from committed reports.\n"
        "% Macro index (oracle F1-max is a reference only):\n"
        + "".join("% \\" + name + "\n" for name in macros)
        + "".join("\\newcommand{\\" + name + "}{" + value + "}\n" for name, value in macros.items()))

    def write(name, spec, headers, rows):
        (tables / (name + ".tex")).write_text(tabular(spec, headers, rows))

    overview, areas = md_tables("reports/dataset/dataset_stats.md")[:2]
    stats = csv_rows("reports/dataset/dataset_stats.csv")
    rows = []
    for cat in CATEGORIES:
        r = next(r for r in overview if r["category"] == cat)
        area = next(r for r in areas if r["category"] == cat)
        types = [r for r in stats if r["category"] == cat and r["defect_type"] != "good"]
        assert len(types) == int(r["defect types"])
        assert sum(int(t["n_test"]) for t in types) == int(r["test defect"])
        rows.append([cat, "obj." if r["kind"] == "object" else "tex.", r["fit / val"],
                     r["test good"] + "/" + r["test defect"], str(len(types)),
                     area["area median"].rstrip("%"), r["pix_std"], r["align_corr"]])
    write("tab_dataset", r"@{}l@{\,}l@{\,}r@{\,}r@{\,}r@{\,}r@{\,}r@{\,}r@{}",
          [r"Category & Kind & Fit/val & \shortstack{Test\\good/def.} & \shortstack{\#\\types} & \shortstack{Area\\(\%)} & \shortstack{Pixel\\std.} & \shortstack{Align.\\corr.} \\"], rows)

    main_metrics = ("ImgAuroc", "ImgFone", "ImgFmax", "PixAuroc", "Aupro")
    rows = []
    for name, cats in [(c, (c,)) for c in CATEGORIES] + [("Mean (obj.)", CATEGORIES[:3]), ("Mean (tex.)", CATEGORIES[3:]), ("Mean (all)", CATEGORIES)]:
        if name == "Mean (obj.)":
            rows.append(None)
        cells = [name]
        for metric in main_metrics:
            cells.extend(pair(*(main.get(run, METRICS[metric], cats) for run in RUNS.values())))
        rows.append(cells)
    write("tab_main", "@{}l" + "@{\\hspace{4pt}}r" * 10 + "@{}",
          [r"& \multicolumn{2}{c}{Image AUROC} & \multicolumn{2}{c}{Image F1 (val thr)} & \multicolumn{2}{c}{\oracle} & \multicolumn{2}{c}{Pixel AUROC} & \multicolumn{2}{c}{AU-PRO} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
           "Category & " + " & ".join(["PaDiM & PatchCore"] * 5) + r" \\"], rows)

    cost = ("Fit", "Infer", "InferEtoE", "Mem")
    values = [[main.get(run, METRICS[m]) for m in cost] for run in RUNS.values()]
    rows = [[label, *[bold(f"{val:.2f}", val == min(values[0][i], values[1][i])) for i, val in enumerate(vals)]]
            for label, vals in zip(("PaDiM", "PatchCore"), values)]
    write("tab_cost", "@{}lrrrr@{}", [r"Method & \shortstack{Fit\\(s)} & \shortstack{Infer\\(ms/img)} & \shortstack{E2E\\(ms/img)} & \shortstack{GPU\\(GB)} \\"], rows)

    rows = []
    calibration = csv_rows(FAILURES + "calibration.csv")
    for method in ("PaDiM", "PatchCore"):
        for cat in ("screw", "tile"):
            r = next(r for r in calibration if r["method"] == method and r["category"] == cat)
            rows.append([method + "--" + cat, r["n_val"], *[f"{float(r[k]):.3f}" for k in
                         ("val_median", "val_max", "image_thr", "thr_over_val_median", "test_anom_median", "precision", "recall", "image_f1")]])
    write("tab_calibration", "@{}l" + "@{\\hspace{4pt}}r" * 9 + "@{}",
          [r"Method--category & $n_{\rm val}$ & \shortstack{Val\\median} & \shortstack{Val\\max} & \shortstack{Image\\thr} & \shortstack{Thr/val\\median} & \shortstack{Anomaly\\median} & Prec. & Recall & F1 \\"], rows)

    region_rows = csv_rows(FAILURES + "region_detection_pooled.csv")
    write("tab_regionsize", "@{}lrrr@{}", [r"Region size & $n$ & PaDiM & PatchCore \\"],
          [[size_label(r["size_bin"]), r["n"], *pair(float(r["PaDiM"]), float(r["PatchCore"]))] for r in region_rows])
    overlap = csv_rows(FAILURES + "miss_overlap.csv")
    rows = []
    for label, key in (("Missed by both", "both_missed"), ("Only PaDiM", "padim_only"), ("Only PatchCore", "patchcore_only")):
        vals = [sum(int(r[key]) for r in overlap if r["category"] == c) for c in CATEGORIES]
        rows.append([label, *map(str, vals), str(sum(vals))])
    write("tab_missoverlap", r"@{}l@{\,}r@{\,}r@{\,}r@{\,}r@{\,}r@{\,}r@{\,}r@{}",
          [r"Missed & Bot. & Cap. & Scr. & Car. & Tile & Lea. & Total \\"], rows)

    fps = csv_rows(FAILURES + "false_positives.csv")
    loc = csv_rows(FAILURES + "localisation.csv")
    rows = []
    for cat in CATEGORIES:
        fp = [next(r for r in fps if r["category"] == cat and r["method"] == m) for m in ("PaDiM", "PatchCore")]
        ls = [next(r for r in loc if r["category"] == cat and r["method"] == m) for m in ("PaDiM", "PatchCore")]
        rates = [int(r["n_fp"]) / int(r["n_good"]) for r in fp]
        rows.append([cat, *[bold(r["n_fp"] + "/" + r["n_good"], rate == min(rates)) for r, rate in zip(fp, rates)],
                     *pair(*(float(r["median_pixel_iou"]) for r in ls)),
                     *pair(*(float(r["peak_outside_gt"]) for r in ls), lower=True)])
    write("tab_fp_loc", "@{}l" + "@{\\hspace{3pt}}r" * 6 + "@{}",
          [r"& \multicolumn{2}{c}{FP/normal} & \multicolumn{2}{c}{TP median IoU} & \multicolumn{2}{c}{Peak outside GT} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Category & PD & PC & PD & PC & PD & PC \\"], rows)

    rows, previous = [], None
    for v in VARIANTS:
        if previous is not None and previous != v.family:
            rows.append(None)
        cells = [v.label, r"post-hoc\posthoc" if v.posthoc else "a priori"]
        for metric in BONUS_METRICS:
            val = variant_value(v, metric, data)
            candidates = [variant_value(w, metric, data) for w in VARIANTS if not w.posthoc]
            best = (min if metric in ("Infer", "Mem") else max)(candidates)
            dp = 2 if metric in ("Infer", "Mem") else 3
            # Compare at displayed precision so indistinguishable ties are bold.
            cells.append(bold(f"{val:.{dp}f}", not v.posthoc and f"{val:.{dp}f}" == f"{best:.{dp}f}"))
        rows.append(cells)
        previous = v.family
    write("tab_bonus", "@{}ll" + "@{\\hspace{5pt}}r" * 6 + "@{}",
          [r"Variant & Kind & \shortstack{Image\\AUROC} & \shortstack{Image F1\\(val thr)} & \shortstack{Pixel\\AUROC} & AU-PRO & \shortstack{Infer\\ms/img} & \shortstack{Peak\\GiB} \\"], rows)

    rows = []
    recall = csv_rows(FAILURES + "recall_by_defect_type.csv")
    for cat in CATEGORIES:
        if rows:
            rows.append(None)
        for r in recall:
            if r["category"] == cat and r["defect_type"] != "good (FPR)":
                rows.append([cat, tex_escape(r["defect_type"]), r["n"], *pair(float(r["PaDiM"]), float(r["PatchCore"]))])
    write("tab_app_recall_type", r"@{}l@{\hspace{4pt}}l@{\hspace{4pt}}rrr@{}",
          [r"Category & Defect & $n$ & PaDiM & PatchCore \\"], rows)

    for family in ("dinov2", "multiscale", "postproc"):
        rows = []
        variants = [VARIANTS[0], *[v for v in VARIANTS if v.family == family]]
        for cat in CATEGORIES:
            if rows:
                rows.append(None)
            for v in variants:
                # All appendix baseline rows use the same main-run baseline.
                vals = [variant_value(v, m, data, (cat,)) for m in ("ImgAuroc", "ImgFone", "Aupro")]
                rows.append([cat, v.label, *[f"{val:.3f}" for val in vals]])
        write("tab_app_bonus_" + family, "@{}llrrr@{}",
              [r"Category & Variant & \shortstack{Image\\AUROC} & \shortstack{Image F1\\(val thr)} & AU-PRO \\"], rows)


def size_label(label: str) -> str:
    return label.replace(">=", r"$\geq$ ").replace("<", "$<$ ").replace("-", "--")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated")
    generate(parser.parse_args().output_dir)
