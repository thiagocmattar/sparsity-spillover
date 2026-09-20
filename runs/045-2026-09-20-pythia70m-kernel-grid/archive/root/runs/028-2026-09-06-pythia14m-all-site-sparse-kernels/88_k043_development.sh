#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run028_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run028_dir/runtime/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
cd "$run028_dir/../.."
"$run028_dir/runtime/venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete" and r["completed"]==105' "$run028_dir/artifacts/final-matrix-001/result.json"
timeout --signal=TERM --kill-after=30s 600 "$run028_dir/runtime/venv/bin/python" -u "$run028_dir/86_k043_joint_probe.py" --attempt k043-joint-001 > "$run028_dir/runtime/k043-joint-001.log" 2>&1
for run028_condition in c01 c11 c25 c30; do
  timeout --signal=TERM --kill-after=30s 600 "$run028_dir/runtime/venv/bin/python" -u "$run028_dir/87_k043_comparison.py" --condition "$run028_condition" --attempt "k043-$run028_condition-graph-001" > "$run028_dir/runtime/k043-$run028_condition-graph-001.log" 2>&1
done
