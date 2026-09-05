#!/usr/bin/env bash
set -euo pipefail
run025_dir=/workspace/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
run025_venv="$run025_dir/artifacts/runtime/venv-pythia"
export PATH="$run025_venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_dir/artifacts/memcheck-002.exit-code"' EXIT
ninja --version
nvcc --version
timeout --signal=TERM --kill-after=30s 900s compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$run025_venv/bin/python" -u "$run025_dir/01_calibrate.py" --primitive-only --attempt rtx5090-memcheck-002 --seconds 900
