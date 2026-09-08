#!/usr/bin/env bash
# Launch only after approval. Run from the transferred project root on the pod.
set -euo pipefail
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
run_dir=runs/031-2026-09-08-signed-activation-density
python_bin=${RUN031_PYTHON:-/workspace/run031/venv/bin/python}
mode=${1:?Use smoke or full}
case "$mode" in
  smoke)
    output="$run_dir/artifacts/attempts/001-runpod-smoke"
    args=(--keys 14M_A0_None 14M_A4-OL1_0.5 14M_A7-OL1_0.5 --max-blocks 8)
    ;;
  full)
    output="$run_dir/artifacts/attempts/002-runpod-full"
    args=()
    ;;
  *) echo 'Use smoke or full' >&2; exit 2 ;;
esac
mkdir -p "$run_dir/artifacts/logs"
# Do not reuse an executed attempt; outputs are immutable on restart.
test ! -e "$output"
setsid timeout --signal=TERM --kill-after=30s 30m "$python_bin" -u "$run_dir/02_evaluate.py" \
  --output "$output" "${args[@]}" > "$run_dir/artifacts/logs/$mode.log" 2>&1 < /dev/null &
echo "$!" > "$run_dir/artifacts/logs/$mode.pid"
echo "Started $mode PID $(cat "$run_dir/artifacts/logs/$mode.pid")"
