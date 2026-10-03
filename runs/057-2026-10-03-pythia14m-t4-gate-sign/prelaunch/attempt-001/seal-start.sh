set -eu
cd /workspace/sparsity-spillover
run=/workspace/sparsity-spillover/runs/057-2026-10-03-pythia14m-t4-gate-sign
test -f /workspace/run057-control/completed
test ! -e "$run/transfer/training-001.tar.gz"
nohup setsid /workspace/run057-venv/bin/python -u "$run/15_seal_training.py" > /workspace/run057-control/seal.log 2>&1 < /dev/null &
echo $! > /workspace/run057-control/seal.pid
cat /workspace/run057-control/seal.pid
