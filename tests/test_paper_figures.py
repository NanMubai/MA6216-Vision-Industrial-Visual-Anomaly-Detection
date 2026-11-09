"""Figure integrity, exact count intervals, and deterministic illustration picks."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_figures
from paper_figures import (CHARTS, EXPECTED_BINS, region_counts, wilson, output_root,
                           log_box_stats, csv_rows, PER_REGION, CATEGORIES)
from paper_panels import PANEL_OUTPUTS, select_examples, gt_contour, place_decision_label, case_label


def test_v1_checksums_and_legacy_figures():
    archive = ROOT / "paper/generated/figures/v1"
    entries = (archive / "SHA256SUMS").read_text().splitlines()
    assert len(entries) == 14
    for line in entries:
        digest, relative = line.split(maxsplit=1)
        assert hashlib.sha256((archive / relative).read_bytes()).hexdigest() == digest, relative
    for name in ["fig_bars.pdf", *[p.name for p in archive.glob("*.jpg")]]:
        assert (archive / name).read_bytes() == (archive.parent / name).read_bytes()


def test_archive_output_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        output_root(tmp_path / "v1")


@pytest.mark.parametrize("k,n", [(27, 50), (17, 50), (0, 50), (50, 50), (84, 88)])
def test_wilson_independent_score_quadratic(k, n):
    # Invert (k - n*p)^2 = z^2*n*p*(1-p), independently of the
    # centre/half-width implementation used by the plotting script.
    z2 = 1.959964**2
    roots = np.sort(np.roots([n + z2, -(2*k + z2), k*k/n]))
    assert wilson(k, n) == pytest.approx(roots, abs=1e-12)
    if (k, n) == (27, 50):
        # Corrected by the coordinator: the original brief's pair was a typo.
        assert tuple(round(v, 4) for v in wilson(k, n)) == (0.4040, 0.6703)


def test_region_bin_counts_match_pooled():
    pooled, counts = region_counts()
    assert tuple(r["size_bin"] for r in pooled) == EXPECTED_BINS
    assert counts["PatchCore"][0] == (27, 50)
    assert counts["PaDiM"][0] == (17, 50)
    for method in counts:
        assert sum(n for _, n in counts[method]) == 616


def test_selection_is_deterministic_and_breaks_ties_by_index():
    pc = np.array([2, 3, 4, 4, 5, 6, 1.2, 1.5, 1.5])
    pd = np.array([0, 12, 0, 9, 5, 6, 0, 0, 0])
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
    first = select_examples(pd, pc, labels, 2, 1)
    second = select_examples(pd.copy(), pc.copy(), labels.copy(), 2, 1)
    assert first == second == [2, 0, 1, 7]


def test_contour_retains_crop_pixel_coordinates():
    mask = np.zeros((224, 224), dtype=bool)
    mask[10:20, 30:40] = True
    fig, ax = plt.subplots()
    gt_contour(ax, mask)
    vertices = ax.collections[-1].get_paths()[0].vertices
    # Pixel outlines must align with the imshow centres, without a vertical flip.
    assert vertices.min(axis=0) == pytest.approx([29.5, 9.5])
    assert vertices.max(axis=0) == pytest.approx([39.5, 19.5])
    plt.close(fig)


def test_decision_label_avoids_occupied_corner():
    gt = np.zeros((224, 224), dtype=bool)
    gt[140:, :100] = True
    fig, ax = plt.subplots(figsize=(2, 2), layout="constrained")
    ax.imshow(gt, interpolation="none")
    label = ax.text(.025, .025, "0.89 · TN", transform=ax.transAxes,
                    fontsize=6, ha="left", va="bottom", bbox={"pad": .7})
    fig.canvas.draw()
    placement = place_decision_label(ax, label, gt, np.zeros_like(gt), fig.canvas.get_renderer())
    assert placement["corner"] != "bottom_left"
    assert placement["candidate_overlap_pixels"]["bottom_left"] > 0
    assert placement["overlap_pixels"] == min(placement["candidate_overlap_pixels"].values()) == 0
    plt.close(fig)


def test_qualitative_header_omits_repeated_category():
    case = dict(letter="a", category="screw", key="manipulated_front/016.png")
    assert case_label(case, qualitative=True) == "(a) manipulated_front 016"
    assert "screw" in case_label(case)


def test_tradeoff_control_stays_above_visible_baseline(monkeypatch, tmp_path):
    figures = []
    monkeypatch.setattr(paper_figures, "save", lambda fig, path: figures.append(fig))
    monkeypatch.setattr(paper_figures, "record", lambda *args, **kwargs: None)
    with plt.style.context(paper_figures.STYLE):
        paper_figures.tradeoff(tmp_path, paper_figures.load_metrics())
    fig, = figures
    for ax in fig.axes:
        baseline, = [line for line in ax.lines if line.get_marker() == "*"]
        control, = [line for line in ax.lines if line.get_marker() == "+"]
        assert baseline.get_markersize() >= 6
        assert control.get_zorder() > baseline.get_zorder()
        assert control.get_markersize() == 7
        assert control.get_markeredgewidth() == 1
        assert control.get_path_effects()  # White understroke separates coincident glyphs.
    assert "max-score control" in [text.get_text() for text in fig.legends[0].get_texts()]
    plt.close(fig)


def test_panel_tiles_are_embedded_at_native_resolution():
    folder = ROOT / "paper/generated/figures"
    for name in PANEL_OUTPUTS:
        listing = subprocess.check_output(["pdfimages", "-list", str(folder / name)], text=True)
        tiles = []
        for row in listing.splitlines()[2:]:
            fields = row.split()
            if fields[2] == "image" and min(int(fields[3]), int(fields[4])) > 100:
                tiles.append(tuple(map(int, fields[3:5])))
                assert fields[9] == "no", (name, row)
        assert tiles and set(tiles) == {(224, 224)}, name


def test_log_space_box_whiskers_use_extreme_observations_inside_fences():
    rows = csv_rows(PER_REGION)
    groups = [[float(r["area_cells"]) for r in rows if r["method"] == "PatchCore" and r["category"] == cat]
              for cat in CATEGORIES]
    groups.append([1, 1, 1, 2, 4, 8, 16, 256])  # Log and linear fences differ.
    for values in groups:
        logged = np.log10(values)
        q1, q3 = np.quantile(logged, [.25, .75])
        inside = np.asarray(values)[(logged >= q1 - 1.5 * (q3 - q1)) & (logged <= q3 + 1.5 * (q3 - q1))]
        stats = log_box_stats(values)
        assert stats["whislo"] == min(inside)
        assert stats["whishi"] == max(inside)
        assert np.log10(stats["q1"]) == pytest.approx(q1)
        assert np.log10(stats["q3"]) == pytest.approx(q3)
    assert log_box_stats(groups[-1])["whishi"] == 256


def test_manifest_and_physical_dimensions():
    folder = ROOT / "paper/generated/figures"
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    assert manifest["minimum_text_pt"] >= 6
    assert set(manifest["outputs"]) == set(CHARTS) | set(PANEL_OUTPUTS)
    total = 0
    for name, entry in manifest["outputs"].items():
        assert "v1" not in name
        assert entry["sources"] and entry["transformation"] and entry["n"]
        assert "selection_rule" in entry and entry["generator"].startswith("scripts/paper_")
        path = folder / name
        total += path.stat().st_size
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        width, height = map(float, re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", info).groups())
        assert width / 72 == pytest.approx(entry["width_inches"], abs=.001)
        assert height / 72 == pytest.approx(entry["height_inches"], abs=.001)
        assert entry["width_inches"] in (3.5, 5.6, 7.16)
        if name in PANEL_OUTPUTS:
            assert path.stat().st_size <= 2_500_000
            assert entry["normalization"] == {"vmin": .4, "vcenter": 1., "vmax": 1.6, "cmap": "RdBu_r", "alpha": .6}
            assert entry["raster_interpolation"] == "none"
            assert entry["image_composite_image"] is False
            for pick in entry["picks"]:
                for method in pick["methods"].values():
                    label = method["decision_label"]
                    assert label["overlap_pixels"] == min(label["candidate_overlap_pixels"].values())
                    assert label["overlap_pixels"] == label["candidate_overlap_pixels"][label["corner"]]
            assert entry["layout"]["image_rows"] == 3
            if name.startswith("fig_app_qual_"):
                assert entry["width_inches"] == 5.6 and entry["height_inches"] <= 3.45
                assert entry["layout"]["case_columns"] == 4
                assert [p["letter"] for p in entry["picks"]] == ["a", "b", "c", "d"]
            else:
                assert entry["height_inches"] <= 3.9
                assert entry["layout"]["case_columns"] == 6
    assert total <= 15_000_000
    assert [p["id"] for p in manifest["outputs"]["fig_failures.pdf"]["picks"]] == ["01", "04", "05", "07", "08", "12"]


def test_tradeoff_points_match_number_macros():
    folder = ROOT / "paper/generated"
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([\d.]+)\}", (folder / "numbers.tex").read_text()))
    manifest = json.loads((folder / "figures/MANIFEST.json").read_text())
    points = manifest["outputs"]["fig_tradeoff.pdf"]["variants"]
    assert len(points) == 9 and not any(k.startswith("Pp") for k in points)
    for variant, metrics in points.items():
        for metric, value in metrics.items():
            dp = 2 if metric == "Infer" else 4
            assert f"{value:.{dp}f}" == macros["bn" + variant + metric]


def test_real_selection_manifest_can_be_reproduced():
    """Read small score arrays only; never load models or run predictions."""
    results = Path("/home/eva/code/ma6216/industrial-visual-anomaly-detection/results")
    if not results.exists():
        pytest.skip("fixed evaluation outputs unavailable")
    manifest = json.loads((ROOT / "paper/generated/figures/MANIFEST.json").read_text())
    for name in PANEL_OUTPUTS[3:]:
        picks = manifest["outputs"][name]["picks"]
        for cat in dict.fromkeys(p["category"] for p in picks):
            scores, thresholds = [], []
            for run in ("padim_wrn50", "patchcore_wrn50"):
                source = results / run / cat
                with np.load(source / "scores.npz", allow_pickle=False) as arrays:
                    scores.append(arrays["image_scores"])
                    labels = arrays["labels"]
                thresholds.append(json.loads((source / "metrics.json").read_text())["image_threshold"])
            first = select_examples(*scores, labels, *thresholds)
            second = select_examples(*scores, labels, *thresholds)
            assert first == second == [p["index"] for p in picks if p["category"] == cat]
