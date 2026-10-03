repo=/workspace/sparsity-spillover
run="$repo/runs/058-2026-10-03-pythia14m-t2-p2-versus-ph"
control=/workspace/run058-control
test -e "$control/completed"
test ! -e "$control/seal.pid"
cd "$repo"
nohup setsid bash -c "/workspace/run058-venv/bin/python '$run/15_seal_training.py'; code=\$?; printf '%s\n' \"\$code\" > '$control/seal.exit'" > "$control/seal.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$control/seal.pid"
