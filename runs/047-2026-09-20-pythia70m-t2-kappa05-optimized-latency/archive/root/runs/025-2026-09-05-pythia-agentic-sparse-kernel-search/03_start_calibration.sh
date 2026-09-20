#!/usr/bin/env bash
# Detached, bounded batch worker. Does NOT stop billing or contain RunPod credentials.
set -euo pipefail
run025_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run025_python="${1:?Pass the prepared venv Python absolute path}"
run025_attempt="${2:?Pass a NEW attempt identifier}"
run025_seconds="${3:-5400}"
[[ "$run025_attempt" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
[[ "$run025_seconds" =~ ^[0-9]+$ ]] && (( run025_seconds > 0 && run025_seconds <= 6600 )) || exit 2
run025_control="$run025_dir/artifacts/control-$run025_attempt"
if [[ "${RUN025_INNER:-0}" != 1 ]]; then
  mkdir -p -- "$run025_dir/artifacts"
  mkdir -- "$run025_control"  # exclusive: never resume/overwrite a prior attempt
  nohup setsid env RUN025_INNER=1 bash "$0" "$run025_python" "$run025_attempt" "$run025_seconds" \
    > "$run025_control/worker.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run025_control/worker.pid"
  printf 'Started %s; log: %s/worker.log\n' "$run025_attempt" "$run025_control"
  exit 0
fi
finish() {
  code=$?
  printf '%s\n' "$code" > "$run025_control/exit-code.txt"
  date -u +%FT%TZ > "$run025_control/finished-utc.txt"
  exit "$code"
}
trap finish EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
unset RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
nvidia-smi -q > "$run025_control/nvidia-smi-q.txt"
"$run025_python" -m pip freeze > "$run025_control/pip-freeze.txt"
timeout --signal=TERM --kill-after=30s "${run025_seconds}s" \
  "$run025_python" -u "$run025_dir/01_calibrate.py" --attempt "$run025_attempt" --seconds "$run025_seconds"
