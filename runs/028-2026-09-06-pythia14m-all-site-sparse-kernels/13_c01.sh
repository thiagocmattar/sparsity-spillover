#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt k020-c01-dev-001 --condition c01 --candidate k020 --inputs 8 --passes 3 > "$run028_dir/runtime/k020-c01-dev-001.log" 2>&1
timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" --attempt k021-c01-dev-001 --condition c01 --candidate k021 --inputs 8 --passes 3 > "$run028_dir/runtime/k021-c01-dev-001.log" 2>&1
