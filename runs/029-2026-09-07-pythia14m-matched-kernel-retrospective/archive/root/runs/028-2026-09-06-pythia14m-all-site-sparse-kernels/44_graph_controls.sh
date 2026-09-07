#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for run028_candidate in k031 k032; do
  for run028_condition in c25 c30; do
    timeout --signal=TERM --kill-after=30s 900 bash "$run028_dir/37_attribution.sh" --attempt "$run028_candidate-$run028_condition-graph-001" --candidate "$run028_candidate" --condition "$run028_condition" --inputs 32 --passes 7 --execution graph > "$run028_dir/runtime/$run028_candidate-$run028_condition-graph-001.log" 2>&1
  done
done
