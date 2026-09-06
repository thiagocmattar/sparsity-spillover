#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run026_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
cd "$run026_dir/../.."
run026_py="$run026_dir/runtime/venv/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 012-joint-canonical-workspace --idea 'Check composed kernel against the original canonical attention wrapper' --implementation k019 --joint-with-rope --mode native --controls native graph --full-validation --profile
unset CUBLAS_WORKSPACE_CONFIG
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 013-joint-canonical-default --idea 'Check composed kernel against canonical dense with default workspace' --implementation k019 --joint-with-rope --mode native --controls native graph --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 014-joint-hoisted-default --idea 'Isolate import hoisting from the workspace setting' --implementation k019 --joint-with-rope --mode native --controls native graph --hoist-attention-import --full-validation --profile
"$run026_py" "$run026_dir/autoresearch/progress.py"
"$run026_py" -u "$run026_dir/autoresearch/diagnose_graph.py" --name graph-first-divergence
