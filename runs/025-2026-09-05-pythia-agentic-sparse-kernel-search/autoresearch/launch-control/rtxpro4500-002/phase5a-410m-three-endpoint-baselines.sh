#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase5a-410m-three-endpoint-baselines-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_implementation="$1"
  local run025_condition="$2"
  local run025_sites="$3"
  local run025_short="$4"
  local run025_label="$run025_implementation-410m-$run025_short-rtxpro4500-002"
  local run025_probe="$run025_dir/autoresearch/probe_models.py"
  if [[ "$run025_implementation" == k004 ]]; then
    run025_probe="$run025_dir/autoresearch/probe_k004_models.py"
  fi
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 900s python -u \
    "$run025_probe" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites "$run025_sites" --execution eager --inputs 16 --passes 3 --seconds 840 \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

for run025_implementation in p0 k001 k004; do
  run_probe "$run025_implementation" 410m/a0 all a0-all
done
for run025_implementation in p0 k001 k004; do
  run_probe "$run025_implementation" 410m/a4-0p5 active a4-0p5-active
done
for run025_implementation in p0 k001 k004; do
  run_probe "$run025_implementation" 410m/a7-0p5 active a7-0p5-active
done

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
