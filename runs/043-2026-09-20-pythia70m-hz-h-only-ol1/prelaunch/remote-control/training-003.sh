set -euo pipefail
cd /workspace/sparsity-spillover
run=runs/043-2026-09-20-pythia70m-hz-h-only-ol1
control=/workspace/run043-control
python=/workspace/run043-venv/bin/python
export PYTHONPATH="$PWD/$run:$PWD/src"
export HF_HOME=/workspace/run043-hf TOKENIZERS_PARALLELISM=false
"$python" - <<'PY'
import json,time
from pathlib import Path
from run_config import RUN_DIR,condition_specs,load_config
records={}
for threads,folder in [(8,RUN_DIR),(1,RUN_DIR/'prelaunch/thread-calibration-002')]:
 rows=[json.loads((folder/'prelaunch'/('remote-preflight-'+c['id']+'.json')).read_text()) for c in condition_specs(load_config())]
 assert all(r['status']=='passed' for r in rows)
 records[threads]=rows
threads=min(records,key=lambda n:max(r['median_boundary_seconds'] for r in records[n]))
rows=records[threads]
training=max(r['predicted_training_seconds'] for r in rows)
diagnostics=max(sum(r['diagnostic_seconds'].values()) for r in rows)
checkpoint=12*max(r['checkpoint_seconds'] for r in rows)
validation=2*max(r['validation_seconds'] for r in rows)
estimate=training*1.15+diagnostics+checkpoint+validation+1200
deadline=1789929961.205
remaining=deadline-time.time()
assert estimate<remaining,(estimate,remaining)
record=dict(cpu_threads=threads,comparisons={n:[r['median_boundary_seconds'] for r in rs] for n,rs in records.items()},
 predicted_training_seconds=training,measured_diagnostics_seconds=diagnostics,
 estimated_checkpoint_seconds=checkpoint,estimated_validation_seconds=validation,
 training_margin_fraction=.15,archive_and_transfer_reserve_seconds=1200,
 estimated_total_seconds=estimate,deadline_remaining_seconds=remaining,
 deadline_epoch=deadline,scientific_inputs_unchanged=True)
(RUN_DIR/'prelaunch/training-launch-003.json').write_text(json.dumps(record,indent=2)+'\n')
Path('/workspace/run043-control/chosen-cpu-threads').write_text(str(threads))
print(json.dumps(record),flush=True)
PY
export OMP_NUM_THREADS="$(cat "$control/chosen-cpu-threads")"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
touch "$control/preflight-passed"
workers=(hz-h-ol1-kappa-0 hz-h-ol1-kappa-0p01 hz-h-ol1-kappa-0p05 hz-h-ol1-kappa-0p1)
pids=()
for i in 0 1 2 3; do
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
