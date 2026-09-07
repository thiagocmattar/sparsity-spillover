#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$run026_dir/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
unset CUBLAS_WORKSPACE_CONFIG
cd "$run026_dir/../.."
run026_py="$run026_dir/runtime/venv/bin/python"
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 015-final-r1 --idea 'Frozen canonical-reference replication 1' --implementation k019 --joint-with-rope --mode native --controls native --full-validation --final-timing
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 016-final-r2 --idea 'Frozen canonical-reference replication 2' --implementation k019 --joint-with-rope --mode native --controls native --full-validation --final-timing
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 017-final-r3 --idea 'Frozen canonical-reference replication 3' --implementation k019 --joint-with-rope --mode native --controls native --full-validation --final-timing
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 018-final-rope-ablation --idea 'Frozen QKV fusion alone with dense output projections' --implementation k019 --mode native --controls native --full-validation --final-timing
"$run026_py" -u "$run026_dir/autoresearch/benchmark.py" --attempt 019-final-joint-ablation --idea 'Frozen joint sparse residual alone with canonical QKV processing' --implementation k018 --mode native --controls native --full-validation --final-timing
"$run026_py" "$run026_dir/autoresearch/progress.py"
