#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sparsity-spillover
run=runs/046-2026-09-20-pythia70m-hz-h-only-ol1-kappa05
control=/workspace/run046-control
mkdir -p "$control"
export PYTHONPATH="$PWD/$run:$PWD/src"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=16
export HF_HOME=/workspace/run046-hf
if test ! -e "$control/environment-ready"; then
  bash "$run/00_setup_remote.sh" > "$control/setup.log" 2>&1
fi
python=/workspace/run046-venv/bin/python
if test ! -e "$control/cache-ready"; then
  "$python" "$run/06_build_cache_from_hf.py" > "$control/cache.log" 2>&1
  touch "$control/cache-ready"
fi
workers=(hz-h-ol1-kappa-0p5)
while test ! -e "$control/initialization-ready"; do sleep 5; done
pids=()
for i in 0; do
  CUDA_VISIBLE_DEVICES=$i "$python" -u "$run/05_remote_preflight.py" --worker "${workers[$i]}" > "$control/preflight-$i.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
test "$failed" = 0
"$python" - <<'PY'
import json,time,os
from pathlib import Path
from run_config import RUN_DIR,condition_specs,load_config
rows=[json.loads((RUN_DIR/'prelaunch'/('remote-preflight-'+c['id']+'.json')).read_text()) for c in condition_specs(load_config())]
assert all(r['status']=='passed' for r in rows)
remaining=float(os.environ['RUN046_DEADLINE_EPOCH'])-time.time()
estimate=max(r['predicted_training_seconds'] for r in rows)*1.15+1200
assert estimate<remaining,(estimate,remaining)
print(json.dumps({'preflight':'passed','estimated_remaining_seconds':estimate,'deadline_remaining_seconds':remaining}),flush=True)
PY
touch "$control/preflight-passed"
pids=()
for i in 0; do
  CUDA_VISIBLE_DEVICES=$i "$python" -u "$run/02_train.py" --worker "${workers[$i]}" > "$control/training-$i.log" 2>&1 &
  pids+=("$!"); echo "$!" > "$control/training-$i.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
test "$failed" = 0
"$python" "$run/03_verify.py" > "$control/verification.log" 2>&1
touch "$control/training-verified"

"$python" "$run/15_seal_training.py" > "$control/collection.log" 2>&1
touch "$control/training-sealed"
