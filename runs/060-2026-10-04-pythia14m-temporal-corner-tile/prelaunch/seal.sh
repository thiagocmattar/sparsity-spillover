set -eu
run=/workspace/sparsity-spillover/runs/060-2026-10-04-pythia14m-temporal-corner-tile
control=/workspace/run060-control
test -f "$control/completed"
test "$(cat "$control/pipeline.exit")" = 0
test ! -e "$run/transfer/training-001.tar.gz"
nohup bash -c '/workspace/run060-venv/bin/python -u /workspace/sparsity-spillover/runs/060-2026-10-04-pythia14m-temporal-corner-tile/15_seal_training.py; code=$?; printf "%s\n" "$code" > /workspace/run060-control/seal.exit' > "$control/seal.log" 2>&1 < /dev/null &
echo "$!" > "$control/seal.pid"
printf 'Archive sealing launched\n'
