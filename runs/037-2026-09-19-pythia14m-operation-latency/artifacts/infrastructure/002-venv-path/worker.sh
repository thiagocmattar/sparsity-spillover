#!/usr/bin/env bash
cd /workspace/run037
export PATH=/workspace/run037/runtime/venv/bin:$PATH
bash 05_execute.sh 1789829330.832 > artifacts/infrastructure/002-venv-path/execution.log 2>&1
code=$?
printf '%s\n' "$code" > artifacts/infrastructure/002-venv-path/execution.exit
