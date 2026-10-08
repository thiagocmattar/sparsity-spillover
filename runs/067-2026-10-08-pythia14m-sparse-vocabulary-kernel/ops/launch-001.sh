#!/usr/bin/env bash
set -uo pipefail
cd /workspace/run067
run=runs/067-2026-10-08-pythia14m-sparse-vocabulary-kernel
mkdir -p "$run/ops"
exec >"$run/ops/pipeline-001.log" 2>&1
export RUN067_SOURCE_COMMIT=72ff70aba176da41a5af022d112f255d34c4222e
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
date -u '+SETUP_START %Y-%m-%dT%H:%M:%SZ'
bash "$run/00_setup_remote.sh" && bash "$run/03_execute.sh"
run067_exit=$?
date -u '+PIPELINE_END %Y-%m-%dT%H:%M:%SZ'
printf '%s\n' "$run067_exit" > "$run/ops/exit-001.txt.tmp"
mv "$run/ops/exit-001.txt.tmp" "$run/ops/exit-001.txt"
exit "$run067_exit"
