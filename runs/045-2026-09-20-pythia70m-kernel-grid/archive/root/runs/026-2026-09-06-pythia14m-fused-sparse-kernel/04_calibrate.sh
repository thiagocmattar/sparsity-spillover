#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$run026_dir/../.."
export PATH="$run026_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
bash "$run026_dir/01_setup.sh"
run026_py="$run026_dir/runtime/venv/bin/python"
"$run026_py" "$run026_dir/00_package.py" --verify "$run026_dir/bundles/initial-002/inventory.json"
"$run026_py" -u "$run026_dir/autoresearch/primitive.py"
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 001-dense-screen --idea 'Freeze the strongest correct dense execution control' --implementation dense --controls native graph compile_reduce_overhead compile_max_autotune --full-validation --profile
