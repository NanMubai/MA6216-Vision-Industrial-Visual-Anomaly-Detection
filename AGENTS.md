# Agent conventions for this repo

MA6216 Final Project 3: industrial visual anomaly detection on MVTec AD.
PaDiM is the baseline and PatchCore is the main method. The assignment spec is in
`Final_Project_3_Industrial_Visual_Anomaly_Detection.md`.

## Hard rule: no test-label leakage
- Models are fitted on normal `train` images only. They must never read `batch["label"]` or `batch["mask"]`.
- Thresholds and score normalisation use **only** the normal `val` split (10% of train/good). See `ad.threshold`.
- Test labels and masks are read **only** in `ad.metrics`. Do not tune hyper-parameters on test results.
- `f1_max` is an oracle reference value. Always label it as such.

## Environment (shared, do not reinstall)
- Python: use `/home/eva/code/ma6216/industrial-visual-anomaly-detection/.venv/bin/python`. It has torch 2.11+cu128, timm, numpy, scipy, sklearn, opencv, matplotlib, pandas, pytest.
- Run with `PYTHONPATH=src` from your worktree root (pytest already sets `pythonpath=src`).
- Pretrained weights are cached. Set `HF_HUB_OFFLINE=1` (available: `wide_resnet50_2.tv_in1k`, `vit_base_patch14_dinov2.lvd142m`, `vit_small_patch14_dinov2.lvd142m`).
- Data: `MVTEC_ROOT=/home/eva/code/ma6216/industrial-visual-anomaly-detection/data/mvtec_ad`. Categories: bottle, capsule, screw, carpet, tile, leather.
- **GPU (RTX 5080 16 GB) is shared between agents.** Wrap every GPU job in `scripts/gpu_run.sh <cmd>`. It takes a global flock and runs jobs one at a time. CPU-only unit tests don't need it.

## Interfaces (fixed — do not change signatures without the coordinator)
- `ad.base.AnomalyModel`: `fit(loader)` and `predict(loader) -> Prediction(patch_maps (N,h,w), image_scores (N,) | None)`.
  `patch_maps` are RAW and at feature resolution. Upsampling, smoothing and image-score aggregation happen in `ad.scoring`.
- `ad.data.MVTecDataset(category, split, ...)` yields a dict with `image`, `mask`, `label`, `defect_type` and `path`.
- Stub modules hold the contracts in their docstrings: `scoring`, `threshold`, `metrics`, `coreset`, `padim`, `patchcore`, `visualize`, `run`, `aggregate`.
- Configs live in `configs/*.yaml`. The schema is in `ad.config`.

## Git
- Work only on your assigned branch. Do not touch `main` and do not push.
- Only edit the files your task owns. If you need a change elsewhere, ask the coordinator.
- Use small, logical commits with Conventional Commit messages (e.g. `feat(padim): ...`).
- `data/` and `results/` are gitignored. Never commit datasets, weights or large arrays.
- Run `pytest -q` before reporting done, and include the output in your report.
