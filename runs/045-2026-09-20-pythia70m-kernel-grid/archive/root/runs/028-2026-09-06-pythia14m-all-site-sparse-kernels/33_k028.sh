#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run028_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run028_dir/runtime/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
cd "$run028_dir/../.."
for run028_condition in c25 c30; do
  timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt "k028-$run028_condition-validation-001" --candidate k028 --condition "$run028_condition" --inputs 8 --passes 3 --full-validation > "$run028_dir/runtime/k028-$run028_condition-validation-001.log" 2>&1
done
