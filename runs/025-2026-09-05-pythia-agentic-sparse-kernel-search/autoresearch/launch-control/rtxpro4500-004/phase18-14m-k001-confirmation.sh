#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_phase17="$run025_dir/artifacts/phase17-fixed-rmodel-pairs-rtxpro4500-004"
run025_control="$run025_dir/artifacts/phase18-14m-k001-confirmation-rtxpro4500-004"
test -f "$run025_phase17/all-checks-finished-utc.txt"
test "$(cat "$run025_phase17/exit-code.txt")" = 0
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
printf 'repeat\tcondition\torder\timplementation\tlabel\n' > "$run025_control/process-order.tsv"

run_probe() {
  local run025_repeat="$1"
  local run025_condition="$2"
  local run025_order="$3"
  local run025_implementation="$4"
  local run025_short="$5"
  local run025_label="k001fixed-r${run025_repeat}-${run025_short}-${run025_implementation}-rtxpro4500-004"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$run025_repeat" "$run025_condition" "$run025_order" \
    "$run025_implementation" "$run025_label" >> "$run025_control/process-order.tsv"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 600s python -u \
    "$run025_dir/autoresearch/probe_models.py" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites active --execution eager --inputs 16 --passes 5 \
    --seconds 540 --full-validation --attempt "$run025_label" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

conditions=(
  "14m/a4-0p5 14m-a4-0p5"
  "14m/a7-0p5 14m-a7-0p5"
)
for run025_repeat in 1 2 3; do
  for run025_spec in "${conditions[@]}"; do
    read -r run025_condition run025_short <<< "$run025_spec"
    if [[ "$run025_repeat" == 2 ]]; then
      run_probe "$run025_repeat" "$run025_condition" 1 k001 "$run025_short"
      run_probe "$run025_repeat" "$run025_condition" 2 p0 "$run025_short"
    else
      run_probe "$run025_repeat" "$run025_condition" 1 p0 "$run025_short"
      run_probe "$run025_repeat" "$run025_condition" 2 k001 "$run025_short"
    fi
  done
  date -u +%FT%TZ > "$run025_control/repeat-${run025_repeat}-finished-utc.txt"
done
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
