#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase17-fixed-rmodel-pairs-rtxpro4500-004"
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
  local run025_sites="$5"
  local run025_short="$6"
  local run025_program="$7"
  local run025_label="fixed-r${run025_repeat}-${run025_short}-${run025_implementation}-rtxpro4500-004"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$run025_repeat" "$run025_condition" "$run025_order" \
    "$run025_implementation" "$run025_label" >> "$run025_control/process-order.tsv"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 600s python -u \
    "$run025_dir/autoresearch/$run025_program" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites "$run025_sites" --execution eager --inputs 16 --passes 5 \
    --seconds 540 --full-validation --attempt "$run025_label" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

# Each row fixes the checkpoint and therefore canonical R_model while changing
# only the baseline versus searched implementation. The six endpoints were
# predeclared development endpoints; no held-out checkpoint is selected here.
pairs=(
  "14m/a4-0p5 p0 active probe_models.py k013 active probe_k013_final.py 14m-a4-0p5"
  "14m/a7-0p5 p0 active probe_models.py k013 active probe_k013_final.py 14m-a7-0p5"
  "70m/a4-0p5 k009 active probe_k009_models.py k016 active probe_k016_final.py 70m-a4-0p5"
  "70m/a7-0p5 k009 active probe_k009_models.py k016 active probe_k016_final.py 70m-a7-0p5"
  "410m/a4-0p5 k004 active probe_k004_flagged_z_models.py k010 active probe_frozen_final.py 410m-a4-0p5"
  "410m/a7-0p5 k004 active probe_k004_flagged_z_models.py k010 active probe_frozen_final.py 410m-a7-0p5"
)

for run025_repeat in 1 2 3; do
  for run025_spec in "${pairs[@]}"; do
    read -r run025_condition run025_baseline run025_baseline_sites run025_baseline_program \
      run025_winner run025_winner_sites run025_winner_program run025_short <<< "$run025_spec"
    if [[ "$run025_repeat" == 2 ]]; then
      run_probe "$run025_repeat" "$run025_condition" 1 "$run025_winner" \
        "$run025_winner_sites" "$run025_short" "$run025_winner_program"
      run_probe "$run025_repeat" "$run025_condition" 2 "$run025_baseline" \
        "$run025_baseline_sites" "$run025_short" "$run025_baseline_program"
    else
      run_probe "$run025_repeat" "$run025_condition" 1 "$run025_baseline" \
        "$run025_baseline_sites" "$run025_short" "$run025_baseline_program"
      run_probe "$run025_repeat" "$run025_condition" 2 "$run025_winner" \
        "$run025_winner_sites" "$run025_short" "$run025_winner_program"
    fi
  done
  date -u +%FT%TZ > "$run025_control/repeat-${run025_repeat}-finished-utc.txt"
done

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
