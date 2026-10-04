python3 - <<'PY'
from pathlib import Path
import json,time,subprocess
control=Path('/workspace/run062-control')
run=Path('/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph')
exit_path=control/'pipeline.exit'
exit_code=exit_path.read_text().strip() if exit_path.exists() else None
snapshot={'utc_epoch':time.time(),'preflight_passed':(control/'preflight-passed').exists(),
 'completed':(control/'completed').exists(),'pipeline_exit':exit_code}
if exit_code not in (None,'0'):
 snapshot['pipeline_tail']=(control/'pipeline.log').read_text(errors='replace')[-2200:]
 snapshot['worker_logs']={p.name:p.read_text(errors='replace')[-1600:] for p in control.glob('training-*.log')}
snapshot['workers']=json.loads(subprocess.check_output(['/workspace/run062-venv/bin/python',str(run/'04_monitor.py')],text=True))
for row in snapshot['workers']:
 events=run/'artifacts/attempts'/row['attempt']/'events.jsonl'
 validation=[]
 for line in events.read_text().splitlines():
  try: event=json.loads(line)
  except json.JSONDecodeError: continue
  if event.get('event')=='validation': validation.append({k:event[k] for k in ['step','loss']})
 row['last_validation']=validation[-1] if validation else None
scalar_exit=control/'scalar-structure.exit'
snapshot['scalar_structure_exit']=scalar_exit.read_text().strip() if scalar_exit.exists() else None
if (control/'scalar-structure.log').exists(): snapshot['scalar_structure_tail']=(control/'scalar-structure.log').read_text(errors='replace')[-1600:]
print(json.dumps(snapshot,indent=2))
PY
nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 /workspace
