#!/usr/bin/env bash
set -euo pipefail
run027_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run027_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
unset CUBLAS_WORKSPACE_CONFIG
cd "$run027_dir/../.."
python -u "$run027_dir/02_primitives.py"
python -u "$run027_dir/01_benchmark.py" --condition c01 --replicate 1 --attempt smoke-a0 --smoke
python -u "$run027_dir/01_benchmark.py" --condition c30 --replicate 1 --attempt smoke-a7 --smoke
