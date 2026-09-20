#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/03_probe.sh" --attempt k023-real-001 --candidate k023 --conditions c25 c30 --inputs 2 > "$run028_dir/runtime/k023-real-001.log" 2>&1
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt k023-c30-dev-001 --candidate k023 --condition c30 --inputs 8 --passes 3 > "$run028_dir/runtime/k023-c30-dev-001.log" 2>&1
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt k023-c30-round-dev-001 --candidate k023 --condition c30 --inputs 8 --passes 3 --round-p > "$run028_dir/runtime/k023-c30-round-dev-001.log" 2>&1
