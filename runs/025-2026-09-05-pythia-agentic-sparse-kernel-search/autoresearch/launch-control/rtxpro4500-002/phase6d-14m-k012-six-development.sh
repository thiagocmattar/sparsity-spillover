#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase6d-14m-k012-six-development-rtxpro4500-002"
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
  local run025_label="k012dev-14m-$run025_short-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s python -u \
    "$run025_dir/autoresearch/probe_k012_models.py" \
    --condition "$run025_condition" --implementation k012 --sites "$run025_sites" \
    --execution eager --inputs 16 --passes 10 --seconds 600 \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

run_probe 14m/a0 all a0
run_probe 14m/a1h h a1h
run_probe 14m/a4-0 active a4-0
run_probe 14m/a4-0p5 active a4-0p5
run_probe 14m/a7-0 active a7-0
run_probe 14m/a7-0p5 active a7-0p5

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
