#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run048
export RUN048_DEADLINE_EPOCH=1789994643
runtime/venv/bin/python 01_prepare.py verify
runtime/venv/bin/python 09_archive_cache.py
touch runtime/setup-complete
bash 05_execute.sh 06_cuda_checks.py
bash 05_execute.sh 03_execute.py --phase smoke --tag 001 --deadline-epoch 1789994643
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch 1789994643
runtime/venv/bin/python 07_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
