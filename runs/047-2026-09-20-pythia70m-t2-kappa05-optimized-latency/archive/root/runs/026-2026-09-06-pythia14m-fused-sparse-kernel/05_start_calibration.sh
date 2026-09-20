#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$run026_dir/runtime"
if [[ -e "$run026_dir/runtime/calibration.pid" ]]; then
  echo 'PID record already exists; inspect the original process before retry' >&2
  exit 2
fi
setsid timeout --signal=TERM --kill-after=30s 2400s bash "$run026_dir/04_calibrate.sh" > "$run026_dir/runtime/calibration.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run026_dir/runtime/calibration.pid"
cat "$run026_dir/runtime/calibration.pid"
