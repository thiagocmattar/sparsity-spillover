#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/sparsity-spillover
control=/workspace/run019-control
environment=/workspace/run019-cache-venv
builder="$repo/runs/004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py"

mkdir -p "$control"
date -u +%FT%TZ > "$control/cache-build-started-utc.txt"
status=0
{
    set -euo pipefail
    python3.12 -m venv "$environment"
    "$environment/bin/python" -m pip install --no-cache-dir --upgrade \
      pip==25.0.1 setuptools==70.2.0
    "$environment/bin/python" -m pip install --no-cache-dir \
      datasets==5.0.0 numpy==2.5.0 transformers==5.12.1
    export HF_HOME=/workspace/hf-cache
    "$environment/bin/python" "$builder" \
      --splits validation train \
      --batch-size 256 \
      --log-every-documents 10000
} > "$control/cache-build.log" 2>&1 || status=$?

printf '%s\n' "$status" > "$control/cache-build-exit-code.txt"
date -u +%FT%TZ > "$control/cache-build-finished-utc.txt"
exit "$status"

