repo=/workspace/sparsity-spillover
run="$repo/runs/058-2026-10-03-pythia14m-t2-p2-versus-ph"
py=/workspace/run058-venv/bin/python
cd "$repo"
"$py" "$run/04_monitor.py"
python3 - <<'PY'
from pathlib import Path
import json
run=Path('/workspace/sparsity-spillover/runs/058-2026-10-03-pythia14m-t2-p2-versus-ph')
control=Path('/workspace/run058-control')
print(json.dumps({'preflight_passed':(control/'preflight-passed').exists(),'completed':(control/'completed').exists(),
    'pipeline_exit':(control/'pipeline.exit').read_text().strip() if (control/'pipeline.exit').exists() else None}))
for path in sorted((run/'prelaunch').glob('remote-preflight-*.json')):
    p=json.loads(path.read_text())
    print(json.dumps({k:p[k] for k in ('worker','status','median_boundary_seconds','reserved_memory_fraction') if k in p}))
PY
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
df -h /workspace
