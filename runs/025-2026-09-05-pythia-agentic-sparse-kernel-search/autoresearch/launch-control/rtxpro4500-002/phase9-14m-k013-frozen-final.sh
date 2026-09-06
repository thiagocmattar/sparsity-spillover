#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase9-14m-k013-frozen-final-rtxpro4500-002"
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
  local run025_partition="$4"
  local run025_label="k013final-14m-$run025_short-$run025_partition-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 1200s python -u \
    "$run025_dir/autoresearch/probe_k013_final.py" \
    --condition "$run025_condition" --implementation k013 --sites "$run025_sites" \
    --execution eager --inputs 16 --passes 5 --seconds 1140 --full-validation \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
  return "$run025_result"
}

run025_development_failed=0
run_probe 14m/a0 all a0 development || run025_development_failed=1
run_probe 14m/a1h h a1h development || run025_development_failed=1
run_probe 14m/a4-0 active a4-0 development || run025_development_failed=1
run_probe 14m/a4-0p5 active a4-0p5 development || run025_development_failed=1
run_probe 14m/a7-0 active a7-0 development || run025_development_failed=1
run_probe 14m/a7-0p5 active a7-0p5 development || run025_development_failed=1

if [ "$run025_development_failed" -ne 0 ]; then
  printf 'failed-heldout-skipped\n' > "$run025_control/development-gate.txt"
else
  printf 'passed\n' > "$run025_control/development-gate.txt"
  run025_heldout_failed=0
  run_probe 14m/a4-0p01 active a4-0p01 heldout || run025_heldout_failed=1
  run_probe 14m/a4-0p05 active a4-0p05 heldout || run025_heldout_failed=1
  run_probe 14m/a4-0p1 active a4-0p1 heldout || run025_heldout_failed=1
  run_probe 14m/a7-0p01 active a7-0p01 heldout || run025_heldout_failed=1
  run_probe 14m/a7-0p05 active a7-0p05 heldout || run025_heldout_failed=1
  run_probe 14m/a7-0p1 active a7-0p1 heldout || run025_heldout_failed=1
  if [ "$run025_heldout_failed" -eq 0 ]; then
    printf 'passed\n' > "$run025_control/heldout-gate.txt"
  else
    printf 'failed\n' > "$run025_control/heldout-gate.txt"
  fi
fi

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
