#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase6a-14m-70m-policy-confirmation-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_probe="$1"
  local run025_condition="$2"
  local run025_implementation="$3"
  local run025_sites="$4"
  local run025_short="$5"
  local run025_label="confirm-$run025_short-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s python -u \
    "$run025_dir/autoresearch/$run025_probe" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites "$run025_sites" --execution eager --inputs 16 --passes 10 \
    --seconds 600 --attempt "$run025_label" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

run_probe probe_models.py 14m/a0 k001 all 14m-a0-k001
run_probe probe_models.py 14m/a1h k001 h 14m-a1h-k001
run_probe probe_models.py 14m/a4-0 k001 active 14m-a4-0-k001
run_probe probe_models.py 14m/a4-0p5 k001 active 14m-a4-0p5-k001
run_probe probe_models.py 14m/a7-0 k001 active 14m-a7-0-k001
run_probe probe_models.py 14m/a7-0p5 k001 active 14m-a7-0p5-k001

run_probe probe_models.py 70m/a4-0p5 k001 active 70m-a4-0p5-k001
run_probe probe_models.py 70m/a7-0p5 k001 active 70m-a7-0p5-k001
run_probe probe_models.py 70m/a4-0p5 k003 active 70m-a4-0p5-k003
run_probe probe_models.py 70m/a7-0p5 k003 active 70m-a7-0p5-k003
run_probe probe_k009_models.py 70m/a4-0p5 k009 active 70m-a4-0p5-k009
run_probe probe_k009_models.py 70m/a7-0p5 k009 active 70m-a7-0p5-k009

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
