#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-015.exit)" = 0
for candidate in opt053 opt054 opt055 opt056 opt057 opt058 opt059 opt060; do
  bash 05_execute.sh candidates/check_head.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt053 opt054 opt055 opt056 opt057 opt058 opt059 opt060 \
  --tag 016 --deadline-epoch 1789926479 >runtime/development-016.log 2>&1
