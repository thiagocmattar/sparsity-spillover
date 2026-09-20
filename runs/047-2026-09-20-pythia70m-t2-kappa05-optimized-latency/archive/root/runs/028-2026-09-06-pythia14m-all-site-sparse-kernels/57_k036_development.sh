#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for run028_condition in c08 c15 c21 c23 c26 c28; do
  timeout --signal=TERM --kill-after=30s 900 bash "$run028_dir/37_attribution.sh" --attempt "k036-$run028_condition-graph-001" --candidate k036 --condition "$run028_condition" --inputs 32 --passes 7 --execution graph > "$run028_dir/runtime/k036-$run028_condition-graph-001.log" 2>&1
done
