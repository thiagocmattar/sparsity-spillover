#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-h100nvl-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase16-h100-frozen-transfer-h100nvl-001"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime-h100nvl-001/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions_h100nvl_001"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_condition="$1"
  local run025_implementation="$2"
  local run025_sites="$3"
  local run025_label="$4"
  local run025_program="$5"
  shift 5
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 900s python -u \
    "$run025_dir/autoresearch/$run025_program" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites "$run025_sites" --execution "$@" --inputs 16 --passes 5 \
    --seconds 840 --full-validation --attempt "$run025_label" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

# Six predeclared development endpoints per architecture. No interior-kappa
# checkpoint is used for this hardware-transfer confirmation.
sentinels=(
  "14m/a0 p0 all 14m-a0 probe_models.py k013 all probe_k013_final.py"
  "14m/a1h p0 h 14m-a1h probe_models.py k013 h probe_k013_final.py"
  "14m/a4-0 p0 active 14m-a4-0 probe_models.py k013 active probe_k013_final.py"
  "14m/a4-0p5 p0 active 14m-a4-0p5 probe_models.py k013 active probe_k013_final.py"
  "14m/a7-0 p0 active 14m-a7-0 probe_models.py k013 active probe_k013_final.py"
  "14m/a7-0p5 p0 active 14m-a7-0p5 probe_models.py k013 active probe_k013_final.py"
  "70m/a0 p0 all 70m-a0 probe_models.py k016 all probe_k016_final.py"
  "70m/a1h p0 h 70m-a1h probe_models.py k016 h probe_k016_final.py"
  "70m/a4-0 p0 active 70m-a4-0 probe_models.py k016 active probe_k016_final.py"
  "70m/a4-0p5 p0 active 70m-a4-0p5 probe_models.py k016 active probe_k016_final.py"
  "70m/a7-0 p0 active 70m-a7-0 probe_models.py k016 active probe_k016_final.py"
  "70m/a7-0p5 p0 active 70m-a7-0p5 probe_models.py k016 active probe_k016_final.py"
  "410m/a0 p0 all 410m-a0 probe_models.py k010 active probe_frozen_final.py"
  "410m/a1h p0 h 410m-a1h probe_models.py k010 active probe_frozen_final.py"
  "410m/a4-0 p0 active 410m-a4-0 probe_models.py k010 active probe_frozen_final.py"
  "410m/a4-0p5 p0 active 410m-a4-0p5 probe_models.py k010 active probe_frozen_final.py"
  "410m/a7-0 p0 active 410m-a7-0 probe_models.py k010 active probe_frozen_final.py"
  "410m/a7-0p5 p0 active 410m-a7-0p5 probe_models.py k010 active probe_frozen_final.py"
)

# P0 is the minimally adapted Sakana starting point. One complete-validation
# process per sentinel supplies both eager and CUDA-graph comparisons.
for run025_spec in "${sentinels[@]}"; do
  read -r run025_condition run025_p0 run025_p0_sites run025_short run025_p0_program run025_winner run025_winner_sites run025_winner_program <<< "$run025_spec"
  run_probe "$run025_condition" "$run025_p0" "$run025_p0_sites" \
    "transfer-p0-${run025_short}-h100nvl-001" "$run025_p0_program" eager graph
done
date -u +%FT%TZ > "$run025_control/p0-finished-utc.txt"

# The winning policy is frozen before seeing H100 results. Repeat 1 includes
# the CUDA-graph control; repeats 2 and 3 are independent eager processes.
for run025_repeat in 1 2 3; do
  for run025_spec in "${sentinels[@]}"; do
    read -r run025_condition run025_p0 run025_p0_sites run025_short run025_p0_program run025_winner run025_winner_sites run025_winner_program <<< "$run025_spec"
    run025_execution=(eager)
    if [[ "$run025_repeat" == 1 ]]; then
      run025_execution+=(graph)
    fi
    run_probe "$run025_condition" "$run025_winner" "$run025_winner_sites" \
      "transfer-winner-r${run025_repeat}-${run025_short}-${run025_winner}-h100nvl-001" \
      "$run025_winner_program" "${run025_execution[@]}"
  done
  date -u +%FT%TZ > "$run025_control/winner-r${run025_repeat}-finished-utc.txt"
done

# Contribution tests preserve each frozen implementation and change only the
# eligible FFN or attention projection linears; QK/PV remain dense. K010 is
# already a z-only projection policy, so a 410M projection-only probe would be
# identical to its full winner evaluation.
components=(
  "14m/a0 k013 14m-a0"
  "14m/a1h k013 14m-a1h"
  "14m/a4-0 k013 14m-a4-0"
  "14m/a4-0p5 k013 14m-a4-0p5"
  "14m/a7-0 k013 14m-a7-0"
  "14m/a7-0p5 k013 14m-a7-0p5"
  "70m/a1h k016 70m-a1h"
  "70m/a4-0 k016 70m-a4-0"
  "70m/a4-0p5 k016 70m-a4-0p5"
  "70m/a7-0 k016 70m-a7-0"
  "70m/a7-0p5 k016 70m-a7-0p5"
)

for run025_spec in "${components[@]}"; do
  read -r run025_condition run025_implementation run025_short <<< "$run025_spec"
  run_probe "$run025_condition" "$run025_implementation" active \
    "component-ffn-${run025_short}-${run025_implementation}-h100nvl-001" \
    probe_frozen_ffn.py eager
  if [[ "$run025_condition" != "14m/a1h" && "$run025_condition" != "70m/a1h" ]]; then
    run_probe "$run025_condition" "$run025_implementation" active \
      "component-attnproj-${run025_short}-${run025_implementation}-h100nvl-001" \
      probe_frozen_attention_projection.py eager
  fi
done

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
