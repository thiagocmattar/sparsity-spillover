#!/usr/bin/env bash
set -euo pipefail
run027_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run027_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
unset CUBLAS_WORKSPACE_CONFIG
cd "$run027_dir/../.."
python -u "$run027_dir/03_matrix.py"
