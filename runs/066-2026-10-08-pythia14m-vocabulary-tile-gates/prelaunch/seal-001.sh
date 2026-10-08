set -euo pipefail
run=/workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates
test "$(cat /workspace/run066-control/pipeline.exit)" = 0
test ! -e "$run/transfer/training-001.tar.gz"
mkdir -p "$run/transfer"
nohup bash -c '/workspace/run066-venv/bin/python /workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates/15_seal_training.py; code=$?; printf "%s\n" "$code" > /workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates/transfer/seal.exit' > "$run/transfer/seal.log" 2>&1 < /dev/null &
echo 'Detached artifact sealing launched after successful cohort exit'
