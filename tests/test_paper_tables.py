"""Cross-check the paper artifacts against the committed evidence, on CPU."""
import csv
from decimal import Decimal, ROUND_HALF_UP
import io
import os
from pathlib import Path
import re
import subprocess
import sys
from statistics import mean

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "capsule", "screw", "carpet", "tile", "leather")
RUNS = ("padim_wrn50", "patchcore_wrn50")
METRICS = ("image_auroc", "image_f1", "image_f1_max_oracle", "pixel_auroc", "aupro")


def committed(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT).decode()


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    subprocess.run([sys.executable, "scripts/paper_tables.py", "--output-dir", str(out)], cwd=ROOT, check=True)
    env = {**os.environ, "MPLCONFIGDIR": str(out / "mpl-cache")}
    subprocess.run([sys.executable, "scripts/paper_figures.py", "--output-dir", str(out / "figures")], cwd=ROOT, env=env, check=True)
    return out


def macros(generated):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([\d.]+)\}", (generated / "numbers.tex").read_text()))


def main_cells(generated):
    rows = {}
    for line in (generated / "tables/tab_main.tex").read_text().splitlines():
        if " & " not in line or not line.endswith(r"\\"):
            continue
        cells = line.removesuffix(r"\\").strip().split(" & ")
        name = cells[0]
        if name in CATEGORIES or name.startswith("Mean"):
            rows[name] = [re.sub(r"\\textbf\{([\d.]+)\}", r"\1", c) for c in cells[1:]]
    return rows


@pytest.mark.parametrize("name,cats", [(c, (c,)) for c in CATEGORIES] + [
    ("Mean (obj.)", CATEGORIES[:3]), ("Mean (tex.)", CATEGORIES[3:]), ("Mean (all)", CATEGORIES)])
def test_main_cells_match_csv(generated, name, cats):
    values = {(r["run"], r["category"], r["metric"]): float(r["value"])
              for r in csv.DictReader(io.StringIO(committed("reports/tables/metrics_long.csv")))}
    expected = [f"{mean(values[run, cat, metric] for cat in cats):.3f}" for metric in METRICS for run in RUNS]
    assert main_cells(generated)[name] == expected


def test_macro_values_and_failure_counts(generated):
    expected = dict(pcImgAurocAll="0.9874", pdImgAurocAll="0.9525", pcAuproAll="0.9308",
                    bnPyrMeanImgAuroc="0.9903", bnMsLOneTwoThreeAupro="0.9454",
                    pdImgFoneTile="0.2857", nAnomTest="556", nGoodTest="177",
                    pdDetected="343", pcDetected="484", pdFP="8", pcFP="19",
                    missBoth="61", missPdOnly="152", missPcOnly="11",
                    bnPpGlobalZImgFone="0.8898", bnPpGlobalPooledImgFone="0.8685")
    actual = macros(generated)
    assert {key: actual[key] for key in expected} == expected
    text = (generated / "numbers.tex").read_text()
    assert all("% \\" + key + "\n" in text for key in actual)


def test_oracle_and_posthoc_labels(generated):
    assert "oracle" in (generated / "tables/tab_main.tex").read_text()
    no_ln = next(line for line in (generated / "tables/tab_bonus.tex").read_text().splitlines() if "no-LN" in line)
    assert r"\posthoc" in no_ln and "post-hoc" in no_ln
    assert r"\textbf" not in no_ln


def table_cells(generated, name, prefix):
    line = next(line for line in (generated / "tables" / f"{name}.tex").read_text().splitlines()
                if line.startswith(prefix + " & "))
    return [re.sub(r"\\textbf\{([\d.]+)\}", r"\1", c)
            for c in line.removesuffix(r"\\").strip().split(" & ")]


def rounded_ratio(n, d):
    return str((Decimal(n) / Decimal(d)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def test_diagnostic_rates_use_exact_counts(generated):
    region = list(csv.DictReader(io.StringIO(committed("reports/figures/failure_cases/tables/per_region.csv"))))
    selected = [r for r in region if r["method"] == "PaDiM" and r["size_bin"] == "10-30 cells"]
    n_detected = sum(r["region_detected"] == "True" for r in selected)
    assert (n_detected, len(selected)) == (84, 88)
    assert table_cells(generated, "tab_regionsize", "10--30 cells")[2] == rounded_ratio(n_detected, len(selected)) == "0.955"

    images = list(csv.DictReader(io.StringIO(committed("reports/figures/failure_cases/tables/per_image.csv"))))
    selected = [r for r in images if r["method"] == "PaDiM" and r["category"] == "capsule"
                and r["defect_type"] == "faulty_imprint" and r["label"] == "1"]
    n_detected = sum(r["detected"] == "True" for r in selected)
    assert (n_detected, len(selected)) == (10, 22)
    assert table_cells(generated, "tab_app_recall_type", r"capsule & faulty\_imprint")[3] == rounded_ratio(n_detected, len(selected)) == "0.455"

    selected = [r for r in images if r["method"] == "PatchCore" and r["category"] == "tile" and r["label"] == "1"]
    n_detected = sum(r["detected"] == "True" for r in selected)
    assert (n_detected, len(selected)) == (68, 84)
    assert table_cells(generated, "tab_calibration", "PatchCore--tile")[8] == rounded_ratio(n_detected, len(selected)) == "0.810"


def test_main_bolds_displayed_ties(generated):
    rows = (generated / "tables/tab_main.tex").read_text().splitlines()
    for row in rows:
        if not any(row.startswith(name + " & ") for name in (*CATEGORIES, "Mean (obj.)", "Mean (tex.)", "Mean (all)")):
            continue
        cells = row.removesuffix(r"\\").strip().split(" & ")[1:]
        for a, b in zip(cells[::2], cells[1::2]):
            if re.findall(r"\d\.\d{3}", a) == re.findall(r"\d\.\d{3}", b):
                assert r"\textbf" in a and r"\textbf" in b


def test_bonus_highlights_accuracy_without_costs(generated):
    path = generated / "tables/tab_bonus.tex"
    rows = [line.removesuffix(r"\\").strip().split(" & ") for line in path.read_text().splitlines()
            if line.endswith(r"\\") and " & " in line][1:]
    assert rows[0][1] == "--"
    assert next(cells for cells in rows if "no-LN" in cells[0])[1] == "post-hoc"
    for cells in rows:
        assert r"\textbf" not in cells[-1] and r"\textbf" not in cells[-2]
    pixel_best = [cells[0] for cells in rows if r"\textbf" in cells[4]]
    assert pixel_best == ["Layers 1+2+3"]
    assert "GiB" in (generated / "tables/tab_cost.tex").read_text()


@pytest.mark.parametrize("line,fail", [
    ("LaTeX Font Warning: Font shape `TS1/ptm/m/sc' undefined for symbol `textdagger'", False),
    ("LaTeX Warning: Reference `sec:missing' on page 1 undefined on input line 9.", True),
    ("LaTeX Warning: Citation `missing2026' on page 1 undefined on input line 9.", True),
    ("LaTeX Warning: There were undefined references.", True),
    (r"Overfull \hbox (5.0pt too wide) in paragraph at lines 1--2", False),
    (r"Overfull \hbox (5.001pt too wide) in paragraph at lines 1--2", True),
])
def test_make_check_log_filter(tmp_path, line, fail):
    (tmp_path / "main.log").write_text(line + "\n")
    # Mark the prerequisite old: exercise the real check recipe on a fixture log
    # without asking latexmk to replace it with another build.
    result = subprocess.run(["make", "--no-print-directory", "-f", str(ROOT / "paper/Makefile"),
                             "-o", "all", "check", f"PYTHON={sys.executable}"], cwd=tmp_path,
                            text=True, capture_output=True)
    assert bool(result.returncode) == fail, result.stdout + result.stderr
    if fail:
        assert line in result.stdout
    else:
        assert "PASS:" in result.stdout


def test_report_notes_section_5_1_rounding(generated):
    """Check every category/mean and all nine metric pairs, including costs."""
    section = committed("docs/report_notes.md").split("### 5.1 ")[1].split("### 5.2 ")[0]
    macro_metrics = ("ImgAuroc", "ImgFone", "ImgFmax", "PixAuroc", "Aupro", "Fit", "Infer", "InferEtoE", "Mem")
    actual = macros(generated)
    checked = 0
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        name = cells[0].strip("*")
        group = {"mean (objects)": "Obj", "mean (textures)": "Tex", "mean (all)": "All"}.get(name, name.title())
        if name not in CATEGORIES and not name.startswith("mean"):
            continue
        assert len(cells) == 10
        for metric, cell in zip(macro_metrics, cells[1:]):
            for prefix, value in zip(("pd", "pc"), cell.split(" / ")):
                assert actual[prefix + metric + group] == value
                checked += 1
    assert checked == 9 * 9 * 2


def test_committed_outputs_are_current(generated):
    for p in [generated / "numbers.tex", *(generated / "tables").glob("*.tex")]:
        assert p.read_bytes() == (ROOT / "paper/generated" / p.relative_to(generated)).read_bytes()


def test_figure_artifacts(generated):
    figures = generated / "figures"
    assert len(list(figures.glob("*.pdf"))) == 4
    assert len(list(figures.glob("*.jpg"))) == 9
    assert sum(p.stat().st_size for p in figures.iterdir()) < 12_000_000
    assert (figures / "fig_failures.jpg").stat().st_size <= 1_500_000
    for path in figures.glob("*.jpg"):
        with Image.open(path) as im:
            assert im.width > 1900
            assert im.info["dpi"][0] <= 300
