"""Qualitative figures: overlays of predicted anomalous regions vs GT masks.

Owner: branch feat/viz.

Figures are drawn on standalone ``matplotlib.figure.Figure`` objects with the Agg canvas, so this
module never touches pyplot's global state or the interactive backend (safe in workers / headless).
Colour code everywhere: GT contour = green, predicted region (map >= pixel_thr) = red.

Note on data use: ``select_cases`` / ``save_category_figures`` read test labels and masks from the
saved result files. This is post-hoc qualitative analysis only; nothing here feeds back into
fitting, thresholds or hyper-parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

GT_COLOR = "lime"
PRED_COLOR = "red"
HEAT_CMAP = "jet"
HEAT_ALPHA = 0.5
DPI = 150

CASE_ORDER = ("tp_best", "tp_poor_loc", "fn", "fp")
CASE_TITLES = {
    "tp_best": "TP, best localisation",
    "tp_poor_loc": "TP, poor localisation",
    "fn": "FN (missed anomaly)",
    "fp": "FP (false alarm)",
}


# --------------------------------------------------------------------------- drawing helpers
def _new_figure(figsize: tuple[float, float]) -> Figure:
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _contour(ax, mask: np.ndarray, color: str, lw: float = 1.2) -> None:
    m = np.asarray(mask) > 0
    if m.any() and not m.all():
        ax.contour(m.astype(np.float32), levels=[0.5], colors=color, linewidths=lw)


def _heat(ax, image: np.ndarray, amap: np.ndarray, vmin: float, vmax: float):
    ax.imshow(image)
    ax.imshow(amap, cmap=HEAT_CMAP, alpha=HEAT_ALPHA, vmin=vmin, vmax=vmax)
    # opaque mappable for the colorbar (the overlay itself is semi-transparent)
    return ScalarMappable(norm=Normalize(vmin, vmax), cmap=HEAT_CMAP)


def _clean(ax, title: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def _save(fig: Figure, save_path: str | os.PathLike) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")


# --------------------------------------------------------------------------- public API
def overlay_panel(image: np.ndarray, gt_mask: np.ndarray, anomaly_map: np.ndarray,
                  pixel_thr: float, title: str = "", vmin: float | None = None,
                  vmax: float | None = None, save_path: str | None = None):
    """4-panel figure: image | image + GT contour | heatmap overlay | predicted region
    (map >= pixel_thr) contour together with GT contour. image (H,W,3) uint8, gt_mask (H,W)
    {0,1}, anomaly_map (H,W) float. Returns the matplotlib Figure (closed if saved)."""
    image = np.asarray(image)
    gt = np.asarray(gt_mask) > 0
    amap = np.asarray(anomaly_map, dtype=np.float32)
    vmin = float(amap.min()) if vmin is None else float(vmin)
    vmax = float(amap.max()) if vmax is None else float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    pred = amap >= pixel_thr

    fig = _new_figure((12.0, 3.4))
    axes = fig.subplots(1, 4)
    axes[0].imshow(image)
    _clean(axes[0], "image")

    axes[1].imshow(image)
    _contour(axes[1], gt, GT_COLOR)
    _clean(axes[1], "GT (green)")

    im = _heat(axes[2], image, amap, vmin, vmax)
    _clean(axes[2], "anomaly map")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

    axes[3].imshow(image)
    _contour(axes[3], pred, PRED_COLOR, lw=2.0)  # thicker so it stays visible under a matching GT
    _contour(axes[3], gt, GT_COLOR)
    _clean(axes[3], f"pred >= {pixel_thr:.3g} (red) vs GT")

    if title:
        fig.suptitle(title, fontsize=10)
    if save_path is not None:
        _save(fig, save_path)  # never registered with pyplot, so nothing to close / no leak
    return fig


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def select_cases(labels: np.ndarray, image_scores: np.ndarray, image_thr: float,
                 masks: np.ndarray, maps: np.ndarray, pixel_thr: float, k: int = 3) -> dict:
    """Pick indices for qualitative analysis: {"tp_best": [...], "fn": [...], "fp": [...],
    "tp_poor_loc": [...]} (TP with lowest pixel IoU). Up to k each."""
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(image_scores, dtype=np.float64).ravel()
    pred_img = scores >= image_thr

    tp = np.flatnonzero((labels == 1) & pred_img)
    fn = np.flatnonzero((labels == 1) & ~pred_img)
    fp = np.flatnonzero((labels == 0) & pred_img)

    # IoU only for TPs (one image at a time keeps memory flat for large N).
    ious = np.array([_iou(np.asarray(maps[i]) >= pixel_thr, np.asarray(masks[i]) > 0) for i in tp])
    ious = np.nan_to_num(ious, nan=0.0)
    order = np.argsort(-ious, kind="stable")  # best first
    tp_best = tp[order[:k]].tolist()
    # Poor localisation: lowest IoU, distinct from the "best" picks when there are enough TPs.
    rest = [i for i in tp[order[::-1]].tolist() if i not in tp_best]
    tp_poor = rest[:k]

    # FN: most confidently missed (lowest score); FP: most confident false alarm (highest score).
    fn_sel = fn[np.argsort(scores[fn], kind="stable")][:k].tolist()
    fp_sel = fp[np.argsort(-scores[fp], kind="stable")][:k].tolist()

    to_int = lambda xs: [int(x) for x in xs]  # noqa: E731
    return {"tp_best": to_int(tp_best), "fn": to_int(fn_sel), "fp": to_int(fp_sel),
            "tp_poor_loc": to_int(tp_poor)}


def _resolve_path(path: str) -> Path | None:
    """Image path from scores.npz; if it moved, retry under $MVTEC_ROOT/<cat>/test/<defect>/<file>."""
    p = Path(path)
    if p.is_file():
        return p
    from ad.data import get_root

    if len(p.parts) >= 4:
        alt = get_root() / Path(*p.parts[-4:])
        if alt.is_file():
            return alt
    return None


def _load_image(path: str, resize: int, crop: int, out_hw: tuple[int, int]) -> np.ndarray:
    """Same geometry as ``MVTecDataset._geom`` (resize -> centre crop), as uint8 RGB."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    p = _resolve_path(path)
    if p is None:
        return np.full((*out_hw, 3), 127, np.uint8)  # placeholder if the image is not on disk
    img = Image.open(p).convert("RGB")
    img = TF.resize(img, [resize, resize], interpolation=TF.InterpolationMode.BILINEAR)
    img = TF.center_crop(img, [crop, crop])
    if img.size != (out_hw[1], out_hw[0]):
        img = img.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _color_range(maps: np.ndarray, pixel_thr: float) -> tuple[float, float]:
    """Shared colour scale for a category: robust to single hot pixels, always spans pixel_thr."""
    flat = maps[:, ::4, ::4].astype(np.float32).ravel()
    vmin = float(np.quantile(flat, 0.01))
    vmax = float(np.quantile(maps.reshape(len(maps), -1).max(axis=1).astype(np.float32), 0.95))
    vmax = max(vmax, float(pixel_thr) * 1.05 if pixel_thr > 0 else float(pixel_thr) + 1e-6)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def save_category_figures(result_dir: str, k: int = 3) -> list[str]:
    """Read results/<run>/<category>/{scores.npz,maps.npz,metrics.json} (format in ad.run)
    and write selected overlay panels to <result_dir>/figs/. Returns written paths."""
    rdir = Path(result_dir)
    sc = np.load(rdir / "scores.npz", allow_pickle=True)
    mp = np.load(rdir / "maps.npz", allow_pickle=True)
    with open(rdir / "metrics.json") as f:
        meta = json.load(f)

    labels = np.asarray(sc["labels"]).astype(int)
    scores = np.asarray(sc["image_scores"], dtype=np.float64)
    dtypes = [str(x) for x in sc["defect_types"]]
    paths = [str(x) for x in sc["paths"]]
    maps = np.asarray(mp["maps"], dtype=np.float32)
    masks = np.asarray(mp["masks"]).astype(np.uint8)

    image_thr = float(meta["image_threshold"])
    pixel_thr = meta.get("pixel_threshold")
    pixel_note = ""
    if pixel_thr is None:  # fall back to the (validation-derived) image threshold
        pixel_thr, pixel_note = image_thr, " [pixel thr n/a: image thr used]"
    pixel_thr = float(pixel_thr)
    data_cfg = (meta.get("config") or {}).get("data") or {}
    resize = int(data_cfg.get("resize", 256))
    crop = int(data_cfg.get("crop", 224))
    category = (meta.get("config") or {}).get("category") or rdir.name

    cases = select_cases(labels, scores, image_thr, masks, maps, pixel_thr, k=k)
    vmin, vmax = _color_range(maps, pixel_thr)
    hw = maps.shape[1:]
    figdir = rdir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    images: dict[int, np.ndarray] = {}
    written: list[str] = []
    for case in CASE_ORDER:
        for i in cases[case]:
            if i not in images:
                images[i] = _load_image(paths[i], resize, crop, hw)
            pred_iou = _iou(maps[i] >= pixel_thr, masks[i] > 0)
            iou_txt = "" if np.isnan(pred_iou) else f"  IoU={pred_iou:.2f}"
            title = (f"{category} | {CASE_TITLES[case]} | {dtypes[i]} (label={labels[i]}) | "
                     f"score={scores[i]:.3g} vs thr={image_thr:.3g}{iou_txt}{pixel_note}")
            out = figdir / f"{case}_{i}_{_safe(dtypes[i])}.png"
            overlay_panel(images[i], masks[i], maps[i], pixel_thr, title=title,
                          vmin=vmin, vmax=vmax, save_path=str(out))
            written.append(str(out))

    # Summary grid: one row per case type, up to k columns; each cell = heatmap + GT + pred contours.
    ncols = max(1, k)
    fig = _new_figure((2.6 * ncols + 0.8, 2.7 * len(CASE_ORDER)))
    axes = np.atleast_2d(fig.subplots(len(CASE_ORDER), ncols, squeeze=False))
    im = None
    for r, case in enumerate(CASE_ORDER):
        for c in range(ncols):
            ax = axes[r, c]
            _clean(ax)
            if c < len(cases[case]):
                i = cases[case][c]
                im = _heat(ax, images[i], maps[i], vmin, vmax)
                _contour(ax, maps[i] >= pixel_thr, PRED_COLOR, lw=1.8)
                _contour(ax, masks[i], GT_COLOR, lw=1.0)
                ax.set_title(f"#{i} {dtypes[i]}\nscore {scores[i]:.3g}", fontsize=8)
            else:
                ax.axis("off")
                if c == 0:
                    ax.text(0.5, 0.5, "(none)", ha="center", va="center", fontsize=9,
                            transform=ax.transAxes)
            if c == 0:
                ax.text(-0.06, 0.5, CASE_TITLES[case], rotation=90, ha="right", va="center",
                        fontsize=9, transform=ax.transAxes)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="anomaly score")
    fig.suptitle(f"{category}: heatmap, GT (green), prediction >= {pixel_thr:.3g} (red); "
                 f"image thr {image_thr:.3g}", fontsize=10)
    summary = figdir / "summary.png"
    _save(fig, summary)
    fig.clear()
    written.append(str(summary))
    return written
