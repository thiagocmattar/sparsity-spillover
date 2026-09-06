#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase7-frozen-final-matrix-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_condition="$1"
  local run025_implementation="$2"
  local run025_sites="$3"
  local run025_short="$4"
  local run025_label="final-$run025_short-$run025_implementation-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 1200s python -u \
    "$run025_dir/autoresearch/probe_frozen_final.py" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites "$run025_sites" --execution eager --inputs 16 --passes 5 \
    --seconds 1140 --full-validation --attempt "$run025_label" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
  return "$run025_result"
}

run025_group_14m_development() {
  local failed=0
  run_probe 14m/a0 k012 all 14m-a0 || failed=1
  run_probe 14m/a1h k012 h 14m-a1h || failed=1
  run_probe 14m/a4-0 k012 active 14m-a4-0 || failed=1
  run_probe 14m/a4-0p5 k012 active 14m-a4-0p5 || failed=1
  run_probe 14m/a7-0 k012 active 14m-a7-0 || failed=1
  run_probe 14m/a7-0p5 k012 active 14m-a7-0p5 || failed=1
  return "$failed"
}

run025_group_14m_heldout() {
  local failed=0
  run_probe 14m/a4-0p01 k012 active 14m-a4-0p01 || failed=1
  run_probe 14m/a4-0p05 k012 active 14m-a4-0p05 || failed=1
  run_probe 14m/a4-0p1 k012 active 14m-a4-0p1 || failed=1
  run_probe 14m/a7-0p01 k012 active 14m-a7-0p01 || failed=1
  run_probe 14m/a7-0p05 k012 active 14m-a7-0p05 || failed=1
  run_probe 14m/a7-0p1 k012 active 14m-a7-0p1 || failed=1
  return "$failed"
}

run025_group_70m_development() {
  local failed=0
  run_probe 70m/a0 k009 all 70m-a0 || failed=1
  run_probe 70m/a1h k009 h 70m-a1h || failed=1
  run_probe 70m/a4-0 k009 active 70m-a4-0 || failed=1
  run_probe 70m/a4-0p5 k009 active 70m-a4-0p5 || failed=1
  run_probe 70m/a7-0 k009 active 70m-a7-0 || failed=1
  run_probe 70m/a7-0p5 k009 active 70m-a7-0p5 || failed=1
  return "$failed"
}

run025_group_70m_heldout() {
  local failed=0
  run_probe 70m/a4-0p01 k009 active 70m-a4-0p01 || failed=1
  run_probe 70m/a4-0p05 k009 active 70m-a4-0p05 || failed=1
  run_probe 70m/a4-0p1 k009 active 70m-a4-0p1 || failed=1
  run_probe 70m/a7-0p01 k009 active 70m-a7-0p01 || failed=1
  run_probe 70m/a7-0p05 k009 active 70m-a7-0p05 || failed=1
  run_probe 70m/a7-0p1 k009 active 70m-a7-0p1 || failed=1
  return "$failed"
}

run025_group_410m_development() {
  local failed=0
  run_probe 410m/a0 k010 active 410m-a0 || failed=1
  run_probe 410m/a1h k010 active 410m-a1h || failed=1
  run_probe 410m/a4-0 k010 active 410m-a4-0 || failed=1
  run_probe 410m/a4-0p5 k010 active 410m-a4-0p5 || failed=1
  run_probe 410m/a7-0 k010 active 410m-a7-0 || failed=1
  run_probe 410m/a7-0p5 k010 active 410m-a7-0p5 || failed=1
  return "$failed"
}

if run025_group_14m_development; then
  printf 'passed\n' > "$run025_control/14m-development-gate.txt"
  if run025_group_14m_heldout; then
    printf 'passed\n' > "$run025_control/14m-heldout-gate.txt"
  else
    printf 'failed\n' > "$run025_control/14m-heldout-gate.txt"
  fi
else
  printf 'failed-heldout-skipped\n' > "$run025_control/14m-development-gate.txt"
fi

if run025_group_70m_development; then
  printf 'passed\n' > "$run025_control/70m-development-gate.txt"
  if run025_group_70m_heldout; then
    printf 'passed\n' > "$run025_control/70m-heldout-gate.txt"
  else
    printf 'failed\n' > "$run025_control/70m-heldout-gate.txt"
  fi
else
  printf 'failed-heldout-skipped\n' > "$run025_control/70m-development-gate.txt"
fi

if run025_group_410m_development; then
  printf 'passed\n' > "$run025_control/410m-development-gate.txt"
else
  printf 'failed\n' > "$run025_control/410m-development-gate.txt"
fi
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
