import json

import numpy as np
import pytest
from PIL import Image

from ad.visualize import overlay_panel, save_category_figures, select_cases

H = W = 32


def _box(y0, y1, x0, x1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


@pytest.fixture
def synth():
    """8 test images. idx 0-1 normal & low score, 2 normal & HIGH score (FP),
    3 anomalous with perfect map (best TP), 4 anomalous with map on the wrong spot (poor TP),
    5 anomalous with low score (FN), 6 anomalous partial overlap, 7 normal low."""
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.8, 0.05, 0.7, 0.15])
    masks = np.zeros((8, H, W), np.uint8)
    maps = np.zeros((8, H, W), np.float32)
    for i in (3, 4, 5, 6):
        masks[i] = _box(8, 16, 8, 16)
    maps[2] = _box(0, 4, 0, 4) * 1.0            # FP blob
    maps[3] = masks[3] * 1.0                    # perfect
    maps[4] = _box(20, 28, 20, 28) * 1.0        # wrong location -> IoU 0
    maps[6] = _box(8, 16, 12, 20) * 1.0         # half overlap
    return labels, scores, masks, maps


def test_select_cases_picks_obvious(synth):
    labels, scores, masks, maps = synth
    c = select_cases(labels, scores, image_thr=0.5, masks=masks, maps=maps, pixel_thr=0.5, k=1)
    assert c == {"tp_best": [3], "fn": [5], "fp": [2], "tp_poor_loc": [4]}
    assert all(type(i) is int for v in c.values() for i in v)


def test_select_cases_k_and_short_lists(synth):
    labels, scores, masks, maps = synth
    c = select_cases(labels, scores, 0.5, masks, maps, 0.5, k=3)
    assert c["tp_best"] == [3, 6, 4]
    assert c["tp_poor_loc"] == []            # all TPs already used as "best"
    assert c["fn"] == [5] and c["fp"] == [2]  # fewer than k is fine
    c = select_cases(labels, scores, 0.5, masks, maps, 0.5, k=2)
    assert c["tp_best"] == [3, 6] and c["tp_poor_loc"] == [4]


def test_overlay_panel_writes_file(tmp_path, synth):
    _, _, masks, maps = synth
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    out = tmp_path / "sub" / "p.png"
    fig = overlay_panel(img, masks[6], maps[6], 0.5, title="t", vmin=0, vmax=1, save_path=str(out))
    assert out.is_file() and out.stat().st_size > 0
    assert len(fig.axes) >= 4
    # all-zero mask and constant map must not crash (no contour levels)
    overlay_panel(img, masks[0], maps[0], 0.5, save_path=str(tmp_path / "z.png"))
    assert (tmp_path / "z.png").is_file()


def test_save_category_figures(tmp_path, synth):
    labels, scores, masks, maps = synth
    rdir = tmp_path / "results" / "run" / "toy"
    rdir.mkdir(parents=True)
    paths, dtypes = [], []
    rng = np.random.default_rng(1)
    for i, lab in enumerate(labels):
        d = "crack" if lab else "good"
        p = tmp_path / "mvtec" / "toy" / "test" / d / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
        dtypes.append(d)
    paths[0] = str(tmp_path / "missing" / "x.png")  # missing image -> placeholder, no crash
    np.savez(rdir / "scores.npz", image_scores=scores, labels=labels, defect_types=np.array(dtypes),
             paths=np.array(paths), val_image_scores=np.array([0.1, 0.2]))
    np.savez_compressed(rdir / "maps.npz", maps=maps.astype(np.float16), masks=masks)
    meta = {"image_threshold": 0.5, "pixel_threshold": 0.5,
            "config": {"data": {"resize": 36, "crop": H}}}
    (rdir / "metrics.json").write_text(json.dumps(meta))

    written = save_category_figures(str(rdir), k=1)
    names = sorted(p.split("/")[-1] for p in written)
    assert names == ["fn_5_crack.png", "fp_2_good.png", "summary.png",
                     "tp_best_3_crack.png", "tp_poor_loc_4_crack.png"]
    for p in written:
        assert (rdir / "figs" / p.split("/")[-1]).stat().st_size > 0


def test_dataset_stats_script(fake_mvtec, tmp_path):
    import importlib.util
    from pathlib import Path

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "dataset_stats", Path(__file__).resolve().parents[1] / "scripts" / "dataset_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "rep"
    mod.main(["--root", str(fake_mvtec), "--out", str(out), "--categories", "toy", "missing_cat"])
    for f in ("dataset_stats.md", "dataset_stats.csv", "defect_regions_per_image.csv",
              "samples_toy.png", "defect_area_hist.png"):
        assert (out / f).stat().st_size > 0, f
    df = pd.read_csv(out / "dataset_stats.csv").set_index("defect_type")
    assert df.loc["good", "n_test"] == 3 and df.loc["crack", "n_test"] == 2
    assert abs(df.loc["crack", "area_median"] - 100 * 60 / 300**2) < 1e-6
    assert df.loc["crack", "regions_mean"] == 1
    md = (out / "dataset_stats.md").read_text()
    assert "| toy | texture | 300x300 | 20 | 18 / 2 | 3 | 2 | 1 |" in md
