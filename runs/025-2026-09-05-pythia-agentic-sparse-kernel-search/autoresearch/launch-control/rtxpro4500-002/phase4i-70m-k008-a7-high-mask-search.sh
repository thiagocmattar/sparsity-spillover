#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase4i-70m-k008-a7-high-mask-search-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_check() {
  local run025_mask="$1"
  local run025_label="k008-70m-a7-0p5-${run025_mask}-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s env \
    RUN025_K008_MASK="$run025_mask" python -u \
    "$run025_dir/autoresearch/probe_k008_models.py" \
    --condition 70m/a7-0p5 --implementation k008 --sites active \
    --execution eager --inputs 7 --passes 2 --seconds 600 \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

run_check prefix1
run_check prefix2
run_check prefix3
run_check prefix4
run_check prefix5
run_check suffix1
run_check suffix2
run_check suffix3
run_check suffix4
run_check suffix5
run_check even
run_check odd

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
