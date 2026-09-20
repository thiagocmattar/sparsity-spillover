#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run026_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$run026_dir/../.."
run026_py="$run026_dir/runtime/venv/bin/python"
"$run026_py" -u "$run026_dir/autoresearch/primitive_rope.py"
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 008-k019-eager --idea 'Fuse exact rounded RoPE, symmetric gates and QKV layout' --implementation k019 --mode native --controls native graph --hoist-attention-import --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 009-k019-graph --idea 'Replay fused rounded RoPE and QKV gates' --implementation k019 --mode graph --controls native graph --hoist-attention-import --full-validation
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 010-k019-joint-eager --idea 'Compose fused RoPE and QKV gates with joint sparse residual' --implementation k019 --joint-with-rope --mode native --controls native graph --hoist-attention-import --full-validation --profile
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 011-k019-joint-graph --idea 'Replay composed RoPE and joint sparse residual kernels' --implementation k019 --joint-with-rope --mode graph --controls native graph --hoist-attention-import --full-validation
"$run026_py" "$run026_dir/autoresearch/progress.py"
