#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while test ! -e runtime/pipeline-019.exit; do sleep 10; done
test "$(cat runtime/pipeline-019.exit)" = 0
for candidate in opt065 opt066 opt067 opt068 opt069; do
  bash 05_execute.sh candidates/check_head.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt065 opt066 opt067 opt068 opt069 --tag 020 --deadline-epoch 1789926479 >runtime/development-020.log 2>&1
bash 05_execute.sh 29_profile_development.py --candidate opt064 --condition c21 --tag 020 >runtime/profile-opt064-020.log 2>&1
