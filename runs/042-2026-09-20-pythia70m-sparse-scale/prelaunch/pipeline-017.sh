#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-016.exit)" = 0
for candidate in opt061 opt062; do
  bash 05_execute.sh candidates/check_attention.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt061 opt062 \
  --tag 017 --deadline-epoch 1789926479 >runtime/development-017.log 2>&1
