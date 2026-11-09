"""Qualitative figures: overlays of predicted anomalous regions vs GT masks.

Owner: branch feat/viz.
"""

from __future__ import annotations

import numpy as np


def overlay_panel(image: np.ndarray, gt_mask: np.ndarray, anomaly_map: np.ndarray,
                  pixel_thr: float, title: str = "", vmin: float | None = None,
                  vmax: float | None = None, save_path: str | None = None):
    """4-panel figure: image | image + GT contour | heatmap overlay | predicted region
    (map >= pixel_thr) contour together with GT contour. image (H,W,3) uint8, gt_mask (H,W)
    {0,1}, anomaly_map (H,W) float. Returns the matplotlib Figure (closed if saved)."""
    raise NotImplementedError


def select_cases(labels: np.ndarray, image_scores: np.ndarray, image_thr: float,
                 masks: np.ndarray, maps: np.ndarray, pixel_thr: float, k: int = 3) -> dict:
    """Pick indices for qualitative analysis: {"tp_best": [...], "fn": [...], "fp": [...],
    "tp_poor_loc": [...]} (TP with lowest pixel IoU). Up to k each."""
    raise NotImplementedError


def save_category_figures(result_dir: str, k: int = 3) -> list[str]:
    """Read results/<run>/<category>/{scores.npz,maps.npz,metrics.json} (format in ad.run)
    and write selected overlay panels to <result_dir>/figs/. Returns written paths."""
    raise NotImplementedError
