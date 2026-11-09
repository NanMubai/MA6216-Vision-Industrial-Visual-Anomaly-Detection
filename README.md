# MA6216-Vision-Industrial-Visual-Anomaly-Detection

Industrial visual anomaly detection on MVTec AD. **PaDiM** is the baseline and **PatchCore** is the main method. We also evaluate PatchCore enhancements: a DINOv2 backbone, multi-scale fusion, and post-processing changes.

Categories: bottle, capsule, screw (objects) and carpet, tile, leather (textures).

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[dev]'
```

## Data
Fill in the form on the MVTec AD page to get the download links. Then run:
```bash
.venv/bin/python scripts/download_mvtec.py --url-file links.txt --categories bottle capsule screw carpet tile leather
```
MVTec AD is licensed under CC BY-NC-SA 4.0. Please cite Bergmann et al., CVPR 2019 and IJCV 2021.

## Run
```bash
.venv/bin/python -m ad.run --config configs/patchcore_wrn50.yaml
.venv/bin/python -m ad.aggregate
```
