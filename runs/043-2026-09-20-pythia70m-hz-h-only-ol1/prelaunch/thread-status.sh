python3 - <<'PY'
import json,time
from pathlib import Path
root=Path('/workspace/sparsity-spillover/runs/043-2026-09-20-pythia70m-hz-h-only-ol1')
control=Path('/workspace/run043-control')
rows=[]
for i in range(4):
 p=control/f'thread-calibration-002-{i}.log'
 lines=p.read_text(errors='replace').splitlines() if p.exists() else []
 events=[]
 for line in lines:
  try:events.append(json.loads(line))
  except ValueError:pass
 row={'gpu':i,'last_line':lines[-1][-500:] if lines else None}
 boundaries=[e for e in events if e.get('phase')=='preflight_boundary']
 if boundaries:row['boundary']=boundaries[-1]
 if events and events[-1].get('status')=='passed':
  e=events[-1]
  row={k:e[k] for k in ['worker','status','median_boundary_seconds','predicted_training_seconds','peak_reserved_bytes','validation_seconds','diagnostic_seconds']}
 rows.append(row)
print(json.dumps({'epoch':time.time(),'workers':rows,'preflight_passed':(control/'preflight-passed').exists(),'pipeline_exit':(control/'thread-calibration-002.exit').read_text() if (control/'thread-calibration-002.exit').exists() else None},indent=2))
PY
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
