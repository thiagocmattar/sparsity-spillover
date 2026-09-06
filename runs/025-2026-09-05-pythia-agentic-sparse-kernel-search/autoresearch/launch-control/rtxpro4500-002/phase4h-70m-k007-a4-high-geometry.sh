#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase4h-70m-k007-a4-high-geometry-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_check() {
  local run025_config="$1"
  local run025_label="k007-70m-a4-0p5-${run025_config}-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s env \
    RUN025_K007_CONFIG="$run025_config" python -u \
    "$run025_dir/autoresearch/probe_k007_models.py" \
    --condition 70m/a4-0p5 --implementation k007 --sites active \
    --execution eager --inputs 16 --passes 3 --seconds 600 \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

run_check m16n128k32w4
run_check m16n256k32w4
run_check m16n256k32w8
run_check m16n128k64w8
run_check m16n256k64w8
run_check m32n128k32w8

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
