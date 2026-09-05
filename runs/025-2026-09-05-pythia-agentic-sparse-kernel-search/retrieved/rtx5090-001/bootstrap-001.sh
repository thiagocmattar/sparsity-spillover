#!/usr/bin/env bash
set -euo pipefail
run025_dir=/workspace/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
run025_control="$run025_dir/artifacts/bootstrap-rtx5090-001"
mkdir -p "$run025_control"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
timeout --signal=TERM --kill-after=30s 1500s bash "$run025_dir/00_setup_remote.sh" primitive
run025_python="$run025_dir/artifacts/runtime/venv-pythia/bin/python"
timeout --signal=TERM --kill-after=30s 900s compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$run025_python" -u "$run025_dir/01_calibrate.py" --primitive-only --attempt rtx5090-memcheck-001 --seconds 900
