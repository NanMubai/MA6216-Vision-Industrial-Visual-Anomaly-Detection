"""Experiment runner.

Owner: branch feat/pipeline.

CLI:  python -m ad.run --config configs/patchcore_wrn50.yaml [--categories bottle tile]
                       [--max-train 20] [--out results] [--no-figs]

Per category, writes results/<name>/<category>/:
    scores.npz   image_scores (N,), labels (N,), defect_types (N,), paths (N,),
                 val_image_scores (M,)
    maps.npz     maps (N,H,W) float16 post-processed test maps, masks (N,H,W) uint8
    metrics.json ad.metrics.evaluate(...) + image_threshold, pixel_threshold,
                 fit_seconds, infer_ms_per_image, model.stats(), resolved config
    figs/        overlay panels (ad.visualize.save_category_figures)
Thresholds come only from the normal validation split (ad.threshold).
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
