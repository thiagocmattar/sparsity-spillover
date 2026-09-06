#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for run028_condition in c25 c30; do
  timeout --signal=TERM --kill-after=30s 900 bash "$run028_dir/07_trial.sh" \
    --attempt "k025-$run028_condition-validation-001" --candidate k025 \
    --condition "$run028_condition" --inputs 8 --passes 3 --full-validation \
    > "$run028_dir/runtime/k025-$run028_condition-validation-001.log" 2>&1
done
