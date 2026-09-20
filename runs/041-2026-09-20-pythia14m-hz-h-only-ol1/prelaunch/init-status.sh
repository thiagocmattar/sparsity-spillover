python3 - <<'PY'
from pathlib import Path
import json
root=Path('/workspace/sparsity-spillover/runs/041-2026-09-20-pythia14m-hz-h-only-ol1/prelaunch')
for p in root.glob('remote-preflight-*.json'):
 r=json.loads(p.read_text());print(json.dumps({k:r[k] for k in ['worker','initial_parameter_sha256','initialization','schedule_sha256']}))
PY
