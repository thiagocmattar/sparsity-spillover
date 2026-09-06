#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase5c-410m-high-site-isolation-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_site() {
  local run025_site="$1"
  local run025_condition="$2"
  local run025_short="$3"
  local run025_probe="$run025_dir/autoresearch/probe_k004_models.py"
  local run025_sites=h
  if [[ "$run025_site" == z ]]; then
    run025_probe="$run025_dir/autoresearch/probe_k004_z_models.py"
    run025_sites=active
  fi
  local run025_label="k004${run025_site}-410m-$run025_short-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 900s python -u \
    "$run025_probe" --condition "$run025_condition" --implementation k004 \
    --sites "$run025_sites" --execution eager --inputs 16 --passes 5 --seconds 840 \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

run_site h 410m/a4-0p5 a4-0p5
run_site z 410m/a4-0p5 a4-0p5
run_site h 410m/a7-0p5 a7-0p5
run_site z 410m/a7-0p5 a7-0p5

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
