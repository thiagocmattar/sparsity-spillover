#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run028_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run028_dir/runtime/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
cd "$run028_dir/../.."
for run028_condition in c01 c11 c25 c30; do
  "$run028_dir/runtime/venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete" and all(r["qualified"][m] for m in ["selected_graph","no_skip_graph","attention_dense_graph"])' "$run028_dir/artifacts/k050-$run028_condition-graph-001/result.json"
done
for run028_condition in c08 c15 c21 c23 c26 c28; do
  timeout --signal=TERM --kill-after=30s 600 "$run028_dir/runtime/venv/bin/python" -u "$run028_dir/113_k050_comparison.py" --condition "$run028_condition" --attempt "k050-$run028_condition-graph-001" > "$run028_dir/runtime/k050-$run028_condition-graph-001.log" 2>&1
done
