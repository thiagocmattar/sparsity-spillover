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
for run028_condition in c01 c11 c25 c30; do
  timeout --signal=TERM --kill-after=30s 600 "$run028_dir/runtime/venv/bin/python" -u "$run028_dir/100_k047_comparison.py" --condition "$run028_condition" --attempt "k047-$run028_condition-graph-001" > "$run028_dir/runtime/k047-$run028_condition-graph-001.log" 2>&1
done
