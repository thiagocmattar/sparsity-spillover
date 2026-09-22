set -euo pipefail
cd /workspace/run049
export RUN049_DEADLINE_EPOCH=1790109006
set +e
bash 05_execute.sh provenance/recovery-001/continue.py --deadline-epoch 1790109006
code=$?
if test "$code" = 0; then runtime/venv/bin/python provenance/recovery-001/reduce.py; code=$?; fi
runtime/venv/bin/python 08_collect.py --tag 002
exit "$code"
