set -euo pipefail
control=/workspace/run062-control
run=/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph
test "$(cat "$control/pipeline.exit")" = 0
test -e "$control/completed"
test ! -e "$control/scalar-structure.pid"
test ! -e "$run/artifacts/scalar-structure"
test "$(sha256sum "$run/17_scalar_structure.py" | cut -d' ' -f1)" = b5b5ed8c2d67da8b2a80bac49cb84b6cbf0572479f6c49ea358d68c8fc742c75
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
nohup setsid bash -c 'run=/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph; /workspace/run062-venv/bin/python "$run/17_scalar_structure.py"; code=$?; printf "%s\n" "$code" > /workspace/run062-control/scalar-structure.exit' > "$control/scalar-structure.log" 2>&1 < /dev/null &
echo $! > "$control/scalar-structure.pid"
printf 'Detached full-validation scalar structure replay launched\n'
