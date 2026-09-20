#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-012.exit)" = 0
for candidate in opt041 opt042 opt043; do
  bash 05_execute.sh candidates/check_joint.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt041 opt042 opt043 \
  --tag 013 --deadline-epoch 1789926479 >runtime/development-013.log 2>&1
