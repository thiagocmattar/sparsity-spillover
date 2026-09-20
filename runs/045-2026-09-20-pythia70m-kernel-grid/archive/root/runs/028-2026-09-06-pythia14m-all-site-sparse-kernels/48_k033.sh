#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run028_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run028_dir/runtime/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
cd "$run028_dir/../.."
timeout --signal=TERM --kill-after=30s 600 "$run028_dir/runtime/venv/bin/python" -u "$run028_dir/47_joint_tensor_probe.py" --attempt k033-joint-001 > "$run028_dir/runtime/k033-joint-001.log" 2>&1
for run028_condition in c01 c11 c25 c30; do
  timeout --signal=TERM --kill-after=30s 900 bash "$run028_dir/37_attribution.sh" --attempt "k033-$run028_condition-graph-001" --candidate k033 --condition "$run028_condition" --inputs 32 --passes 7 --execution graph > "$run028_dir/runtime/k033-$run028_condition-graph-001.log" 2>&1
done
