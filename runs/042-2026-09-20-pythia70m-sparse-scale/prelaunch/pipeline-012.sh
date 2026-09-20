#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/profile-011.exit)" = 0
for candidate in opt033 opt034 opt035 opt036 opt037 opt038 opt039 opt040; do
  bash 05_execute.sh candidates/check_head.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt033 opt034 opt035 opt036 opt037 opt038 opt039 opt040 \
  --tag 012 --deadline-epoch 1789926479 >runtime/development-012.log 2>&1
