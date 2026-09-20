#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run045
printf '%s  %s\n' b3f581e5dcc8c0aeee264a666ac2b9902e135d7ab5b409dc8bd0879825ea827b /workspace/run045-input-001.tar.gz | sha256sum -c -
tar -xzf /workspace/run045-input-001.tar.gz -C /workspace/run045
while test ! -f /workspace/run045-control/environment.exit; do sleep 10; done
test "$(cat /workspace/run045-control/environment.exit)" = 0
export RUN045_DEADLINE_EPOCH=1789941035
runtime/venv/bin/python 01_prepare.py verify
touch runtime/setup-complete
bash 05_execute.sh 03_execute.py --phase preflight --tag 001 --deadline-epoch 1789941035
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch 1789941035
runtime/venv/bin/python 06_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
