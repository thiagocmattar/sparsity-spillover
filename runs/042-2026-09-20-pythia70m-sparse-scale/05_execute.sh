#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PATH="$HERE/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR="$HERE/runtime/extensions" MAX_JOBS=2
remaining=$((1789926479 - $(date +%s) - 600))
test "$remaining" -gt 0
exec timeout --signal=TERM --kill-after=30 "$remaining" python -u "$@"
