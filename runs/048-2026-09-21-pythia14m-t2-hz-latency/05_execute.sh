#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"
: "${RUN048_DEADLINE_EPOCH:?Set the approved absolute stop deadline}"
export PATH="$TASK_DIR/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR="$TASK_DIR/runtime/extensions" MAX_JOBS=2
remaining=$((RUN048_DEADLINE_EPOCH - $(date +%s) - 1200))
test "$remaining" -gt 0
exec timeout --signal=TERM --kill-after=30 "$remaining" python -u "$@"
