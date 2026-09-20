#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-013.exit)" = 0
for candidate in opt044 opt045 opt046 opt047 opt048 opt049; do
  bash 05_execute.sh candidates/check_projection.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt044 opt045 opt046 opt047 opt048 opt049 \
  --tag 014 --deadline-epoch 1789926479 >runtime/development-014.log 2>&1
