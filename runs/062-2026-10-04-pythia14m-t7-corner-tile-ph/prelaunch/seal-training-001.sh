set -euo pipefail
control=/workspace/run062-control
run=/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph
test "$(cat "$control/pipeline.exit")" = 0
test "$(cat "$control/scalar-structure.exit")" = 0
test -f "$run/artifacts/scalar-structure/c36/activation_statistics.json"
test -f "$run/artifacts/scalar-structure/c39/activation_statistics.json"
/workspace/run062-venv/bin/python "$run/15_seal_training.py"
