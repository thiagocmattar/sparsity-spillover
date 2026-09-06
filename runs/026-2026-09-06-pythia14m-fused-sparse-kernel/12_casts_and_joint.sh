#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run026_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$run026_dir/../.."
run026_py="$run026_dir/runtime/venv/bin/python"
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 005-dense-casts --idea 'Preserve eager BF16 rounding in compiled dense with isolated models' --implementation dense --controls native graph compile_reduce_overhead compile_max_autotune --hoist-attention-import --emulate-precision-casts --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/primitive_joint.py"
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 006-k018-eager --idea 'Joint sparse W2 and Wo with rounded parallel residual epilogue' --implementation k018 --mode native --controls native graph --hoist-attention-import --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 007-k018-graph --idea 'Replay joint sparse residual kernel under CUDA graphs' --implementation k018 --mode graph --controls native graph --hoist-attention-import --full-validation --profile
"$run026_py" "$run026_dir/autoresearch/progress.py"
