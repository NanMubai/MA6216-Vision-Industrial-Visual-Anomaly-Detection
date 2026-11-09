#!/usr/bin/env bash
# Run from the worktree root. Each category releases the global shared GPU lock.
set -euo pipefail
export HF_HUB_OFFLINE=1
export MVTEC_ROOT="${MVTEC_ROOT:-/home/eva/code/ma6216/industrial-visual-anomaly-detection/data/mvtec_ad}"
export PYTHONPATH=src
MULTISCALE_PYTHON=/home/eva/code/ma6216/industrial-visual-anomaly-detection/.venv/bin/python
for variant in patchcore_wrn50 patchcore_ms_baseline_max patchcore_ms_l123 patchcore_ms_l234 patchcore_ms_res_1_1.5_mean patchcore_ms_res_1_1.5_zmean; do
  for category in bottle capsule screw carpet tile leather; do
    scripts/gpu_run.sh "$MULTISCALE_PYTHON" -m ad.run --config "configs/${variant}.yaml" --categories "$category" --no-figs
  done
done
# metrics.json contains the authoritative per-category results. Each single-category
# invocation replaces summary.json; aggregate reads all metrics.json files directly.
MULTISCALE_AGG_TMP="$(mktemp -d /tmp/multiscale-aggregate.XXXXXX)"
"$MULTISCALE_PYTHON" -m ad.aggregate --runs patchcore_wrn50 patchcore_ms_baseline_max patchcore_ms_l123 patchcore_ms_l234 patchcore_ms_res_1_1.5_mean patchcore_ms_res_1_1.5_zmean --out "$MULTISCALE_AGG_TMP/tables"
# Generate away from the shared figures directory to avoid worker collisions.
mkdir -p reports/bonus/multiscale_tables reports/bonus/multiscale_figures
cp "$MULTISCALE_AGG_TMP/tables/"* reports/bonus/multiscale_tables/
cp "$MULTISCALE_AGG_TMP/figures/"* reports/bonus/multiscale_figures/
scripts/gpu_run.sh "$MULTISCALE_PYTHON" -m pytest -q | tee reports/bonus/multiscale_pytest.txt
