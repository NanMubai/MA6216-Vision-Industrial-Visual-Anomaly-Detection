#!/usr/bin/env bash
# Serialise GPU jobs across all worktrees/agents (16 GB card): wait for the lock, then run.
# Usage: scripts/gpu_run.sh <command...>
set -euo pipefail
LOCK="${GPU_LOCK:-/home/eva/code/ma6216/industrial-visual-anomaly-detection/.gpu.lock}"
exec flock "$LOCK" "$@"
