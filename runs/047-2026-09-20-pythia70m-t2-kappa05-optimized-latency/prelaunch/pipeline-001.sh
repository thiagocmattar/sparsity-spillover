#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run047
printf '%s  %s\n' d03af0397fd461d0670529f3e7c3328bd6fe234a20a9ef2c4dc03f32d72881f7 /workspace/run047-input-001.tar.gz | sha256sum -c -
tar -xzf /workspace/run047-input-001.tar.gz -C /workspace/run047
while test ! -f /workspace/run047-control/environment.exit; do sleep 10; done
test "$(cat /workspace/run047-control/environment.exit)" = 0
export RUN047_DEADLINE_EPOCH=1789948600
runtime/venv/bin/python 01_prepare.py verify
runtime/venv/bin/python 09_archive_cache.py
touch runtime/setup-complete
bash 05_execute.sh 03_execute.py --phase preflight --tag 001 --deadline-epoch 1789948600
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch 1789948600
runtime/venv/bin/python 06_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
