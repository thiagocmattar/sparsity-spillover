#!/usr/bin/env bash
# Invocation-only environment fixes; original pinned scientific sources unchanged.
set -euo pipefail
run025_dir=/workspace/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
run025_control="$run025_dir/artifacts/control-h100-bootstrap-001"
if [[ "${RUN025_H100_INNER:-0}" != 1 ]]; then
  run025_seconds=$(( ${1:?Pass the absolute lease deadline in Unix seconds} - $(date -u +%s) - 600 ))
  (( run025_seconds > 0 && run025_seconds <= 3000 ))
  mkdir -p "$run025_dir/artifacts"
  mkdir "$run025_control"
  nohup setsid env RUN025_H100_INNER=1 timeout --kill-after=20s "${run025_seconds}s" bash "$0" > "$run025_control/worker.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run025_control/worker.pid"
  exit 0
fi
finish() { code=$?; printf '%s\n' "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"; exit "$code"; }
trap finish EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
export PATH="$run025_dir/artifacts/u0-h100-control-001/venv/bin:$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
unset RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY
nvidia-smi -q > "$run025_control/nvidia-smi-q.txt"
timeout --kill-after=20s 900s bash "$run025_dir/00_setup_remote.sh" primitive
timeout --kill-after=20s 600s compute-sanitizer --tool memcheck --error-exitcode 99 "$run025_dir/artifacts/runtime/venv-pythia/bin/python" -u "$run025_dir/01_calibrate.py" --primitive-only --attempt h100-memcheck-001 --seconds 600
timeout --kill-after=20s 1800s bash "$run025_dir/05_upstream_control.sh" h100-control-001
