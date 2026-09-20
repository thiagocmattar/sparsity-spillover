#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/03_probe.sh" --attempt k024-real-001 --candidate k024 --conditions c25 c30 --inputs 2 > "$run028_dir/runtime/k024-real-001.log" 2>&1
for run028_condition in c01 c25 c30; do
  timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt "k024-$run028_condition-dev-001" --candidate k024 --condition "$run028_condition" --inputs 8 --passes 3 > "$run028_dir/runtime/k024-$run028_condition-dev-001.log" 2>&1
done
