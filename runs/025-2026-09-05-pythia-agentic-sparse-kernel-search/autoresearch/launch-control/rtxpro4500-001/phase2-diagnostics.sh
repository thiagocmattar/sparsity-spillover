#!/usr/bin/env bash
set -euo pipefail
run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase2-diagnostics-rtxpro4500-001"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
run_check() {
  local run025_label="$1"
  local run025_limit="$2"
  shift 2
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s "$run025_limit" "$@" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}
for run025_condition in a1h a4-0p5 a7-0p5; do
  run_check "occupancy-$run025_condition" 180s python -u "$run025_dir/autoresearch/collect_development_occupancy.py" \
    --condition "14m/$run025_condition" --attempt "occupancy-14m-$run025_condition-rtxpro4500-001" --seconds 150
done
run_check dense-baselines 900s python -u "$run025_dir/autoresearch/dense_probe/probe.py" \
  --condition 14m/a1h --attempt rtxpro4500-all-dense-001 \
  --modes native graph compile_reduce_overhead max_autotune --inputs 16 --passes 3 --seconds 840
run_check p0-numerics 660s python -u "$run025_dir/autoresearch/numerics/diagnose_layers.py" \
  --condition 14m/a1h --blocks 1 --prefix-sweep --attempt a1h-layer-diagnostic-rtxpro4500-001 --seconds 600
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
