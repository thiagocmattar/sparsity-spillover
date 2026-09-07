#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run028_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run028_dir/runtime/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
cd "$run028_dir/../.."
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt k022-c30-round-dev-001 --candidate k022 --condition c30 --inputs 8 --passes 3 --round-p > "$run028_dir/runtime/k022-c30-round-dev-001.log" 2>&1
timeout --signal=TERM --kill-after=30s 600 "$run028_dir/runtime/venv/bin/python" -u "$run028_dir/17_precision.py" --attempt k022-c30-precision-001 --condition c30 --inputs 2 > "$run028_dir/runtime/k022-c30-precision-001.log" 2>&1
