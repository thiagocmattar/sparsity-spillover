#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase4c-70m-diagnostics-and-attention-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_check() {
  local run025_label="$1"
  shift
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s "$@" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

run_k003() {
  local run025_label="$1"
  local run025_condition="$2"
  local run025_sites="$3"
  run_check "$run025_label" python -u "$run025_dir/autoresearch/probe_models.py" \
    --condition "$run025_condition" --implementation k003 --sites "$run025_sites" \
    --execution eager --inputs 16 --passes 3 --seconds 600 --attempt "$run025_label"
}

run_occupancy() {
  local run025_label="$1"
  local run025_condition="$2"
  run_check "$run025_label" python -u "$run025_dir/autoresearch/collect_development_occupancy.py" \
    --condition "$run025_condition" --seconds 120 --attempt "$run025_label"
}

run_attention() {
  local run025_label="$1"
  local run025_condition="$2"
  run_check "$run025_label" python -u "$run025_dir/autoresearch/probe_attention.py" \
    --condition "$run025_condition" --linears none --sites active \
    --execution eager --sdpa-backend flash --inputs 16 --passes 3 --seconds 600 \
    --attempt "$run025_label"
}

run_k003 k003-70m-a1h-h-rtxpro4500-002 70m/a1h h
run_k003 k003-70m-a4-0-active-rtxpro4500-002 70m/a4-0 active
run_k003 k003-70m-a4-0p5-active-rtxpro4500-002 70m/a4-0p5 active
run_k003 k003-70m-a7-0p5-active-rtxpro4500-002 70m/a7-0p5 active

run_occupancy occupancy-70m-a1h-rtxpro4500-002 70m/a1h
run_occupancy occupancy-70m-a4-0p5-rtxpro4500-002 70m/a4-0p5
run_occupancy occupancy-70m-a7-0-rtxpro4500-002 70m/a7-0
run_occupancy occupancy-70m-a7-0p5-rtxpro4500-002 70m/a7-0p5

run_attention k002-70m-a0-attention-rtxpro4500-002 70m/a0
run_attention k002-70m-a7-0-attention-rtxpro4500-002 70m/a7-0
run_attention k002-70m-a7-0p5-attention-rtxpro4500-002 70m/a7-0p5

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
