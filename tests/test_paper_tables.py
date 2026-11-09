"""Cross-check the paper artifacts against the committed evidence, on CPU."""
import csv
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
