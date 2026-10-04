python3 - <<'PY'
from pathlib import Path
import json,time
control=Path('/workspace/run062-control')
run=Path('/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph')
out={'utc_epoch':time.time(),'cache_ready':(control/'cache-ready').exists(),
 'preflight_passed':(control/'preflight-passed').exists(),
 'completed':(control/'completed').exists(),
 'pipeline_exit':(control/'pipeline.exit').read_text().strip() if (control/'pipeline.exit').exists() else None,
 'pipeline_tail':(control/'pipeline.log').read_text(errors='replace')[-2600:] if (control/'pipeline.log').exists() else None}
out['control_logs']={p.name:p.read_text(errors='replace')[-1300:] for p in control.glob('*.log') if p.name!='pipeline.log'}
out['preflights']=[]
for p in (run/'prelaunch').glob('remote-preflight-*.json'):
 d=json.loads(p.read_text())
 out['preflights'].append({'file':p.name,**{k:d.get(k) for k in ['status','gpu','median_boundary_seconds','peak_reserved_bytes','total_memory_bytes','validation','validation_seconds','diagnostic_seconds']}})
print(json.dumps(out))
PY
if test -f /workspace/run062-control/cache-ready; then
 /workspace/run062-venv/bin/python /workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/04_monitor.py
fi
nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 /workspace
