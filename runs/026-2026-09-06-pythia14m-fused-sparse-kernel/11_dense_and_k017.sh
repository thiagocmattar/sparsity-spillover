#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run026_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$run026_dir/../.."
run026_py="$run026_dir/runtime/venv/bin/python"
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 002-dense-hoisted --idea 'Hoisted attention import and deterministic cuBLAS workspace' --implementation dense --controls native graph compile_reduce_overhead compile_max_autotune --hoist-attention-import --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 003-k017-eager --idea 'Fuse h and z gates into exact sparse projections, 4 elements and 4 warps' --implementation k017 --mode native --controls native graph compile_reduce_overhead compile_max_autotune --hoist-attention-import --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 004-k017-graph --idea 'Replay the same fused h/z kernel under CUDA graphs' --implementation k017 --mode graph --controls native graph compile_reduce_overhead compile_max_autotune --hoist-attention-import --full-validation --profile
"$run026_py" "$run026_dir/autoresearch/progress.py"
