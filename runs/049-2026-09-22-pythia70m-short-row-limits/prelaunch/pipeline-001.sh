#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run049
printf '%s  %s\n' 774eb7fa96e09141e21d3deff158dd1a1b09cf660d4114bdc72c4f43c5e0281d /workspace/run049-input-001.tar.gz | sha256sum -c -
tar -xzf /workspace/run049-input-001.tar.gz -C /workspace/run049
while test ! -f /workspace/run049-control/environment.exit; do sleep 10; done
test "$(cat /workspace/run049-control/environment.exit)" = 0
export RUN049_DEADLINE_EPOCH=1790109006
runtime/venv/bin/python 01_prepare.py verify
runtime/venv/bin/python 09_archive_cache.py
touch runtime/setup-complete
set +e
bash 05_execute.sh 03_execute.py --deadline-epoch 1790109006
code=$?
if test "$code" = 0; then runtime/venv/bin/python 06_reduce.py; code=$?; fi
runtime/venv/bin/python 08_collect.py --tag 001
exit "$code"
