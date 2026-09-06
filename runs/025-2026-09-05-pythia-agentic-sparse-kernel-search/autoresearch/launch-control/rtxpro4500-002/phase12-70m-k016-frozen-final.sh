#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase12-70m-k016-frozen-final-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_condition="$1"
  local run025_sites="$2"
  local run025_short="$3"
  local run025_attempt="k016final-70m-$run025_short-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_attempt.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 1200s python -u \
    "$run025_dir/autoresearch/probe_k016_final.py" \
    --condition "$run025_condition" --implementation k016 --sites "$run025_sites" \
    --execution eager --inputs 16 --passes 5 --seconds 1140 \
    --full-validation --attempt "$run025_attempt" \
    > "$run025_control/$run025_attempt.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_attempt.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_attempt.finished-utc.txt"
}

run_probe 70m/a0 all a0-development
run_probe 70m/a1h active a1h-development
run_probe 70m/a4-0 active a4-0-development
run_probe 70m/a4-0p5 active a4-0p5-development
run_probe 70m/a7-0 active a7-0-development
run_probe 70m/a7-0p5 active a7-0p5-development
run_probe 70m/a4-0p01 active a4-0p01-retrospective
run_probe 70m/a4-0p05 active a4-0p05-retrospective
run_probe 70m/a4-0p1 active a4-0p1-retrospective
run_probe 70m/a7-0p01 active a7-0p01-retrospective
run_probe 70m/a7-0p05 active a7-0p05-retrospective
run_probe 70m/a7-0p1 active a7-0p1-retrospective
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
