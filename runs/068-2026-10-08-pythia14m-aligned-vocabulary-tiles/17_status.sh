set -eu
control=/workspace/run068-control
run=/workspace/sparsity-spillover/runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles
date -u '+STATUS %Y-%m-%dT%H:%M:%SZ'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
df -h /workspace
find "$control" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
python3 - <<'PY'
from pathlib import Path
for name in ('pipeline.log','cache-build.log'):
 p=Path('/workspace/run068-control')/name
 if p.exists():
  print(name)
  print('\n'.join(p.read_text(errors='replace').splitlines()[-6:]))
PY
if [ -f "$control/pipeline.exit" ]; then cat "$control/pipeline.exit"; fi
if [ -f "$control/preflight-passed" ]; then
  /workspace/run068-venv/bin/python "$run/04_monitor.py"
  python3 - <<'PY'
import json,math
from pathlib import Path
run=Path('/workspace/sparsity-spillover/runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles')
for path in sorted((run/'artifacts/attempts').glob('*/events.jsonl')):
 rows=[]
 for line in path.read_text().splitlines():
  try: rows.append(json.loads(line))
  except json.JSONDecodeError: pass
 updates=[r for r in rows if r.get('event')=='train']
 warnings=[{'step':r['step'],'skipped':r.get('optimizer_step_skipped'),'overflow':r.get('gradient_overflow')}
           for r in updates if r.get('optimizer_step_skipped') or r.get('gradient_overflow')
           or not math.isfinite(r['task_loss']) or r.get('pressure_capture_tensor_count')!=7
           or r.get('pressure_to_task_ratio_final',0)>1+1e-9]
 validations=[r for r in rows if r.get('event')=='validation']
 print(json.dumps({'attempt':path.parent.name,'checked_boundaries':len(updates),'warnings':warnings,
                   'latest_validation':None if not validations else
                      {k:validations[-1].get(k) for k in ('step','loss','sequences','input_tokens','excluded_tail_tokens')}}))
PY
else
  for f in "$control"/preflight-*.log; do [ ! -f "$f" ] || tail -n 5 "$f"; done
fi
