set -euo pipefail
control=/workspace/run062-control
run=/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph
test "$(cat "$control/pipeline.exit")" = 0
test "$(cat "$control/scalar-structure.exit")" = 0
test -f "$run/artifacts/scalar-structure/c36/activation_statistics.json"
test -f "$run/artifacts/scalar-structure/c39/activation_statistics.json"
test ! -e "$control/seal-training.pid"
nohup setsid bash -c '/workspace/run062-venv/bin/python /workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/15_seal_training.py; code=$?; printf "%s\n" "$code" > /workspace/run062-control/seal-training.exit' > "$control/seal-training.log" 2>&1 < /dev/null &
echo $! > "$control/seal-training.pid"
printf 'Detached training archive sealing launched\n'
