set -eu
date -u
tail -n 2 /workspace/run057-control/cache.log
python3 - <<'PY'
import json
from pathlib import Path
run=Path('/workspace/sparsity-spillover/runs/057-2026-10-03-pythia14m-t4-gate-sign')
for path in sorted((run/'prelaunch').glob('remote-preflight-*.json')):
    row=json.loads(path.read_text())
    print(json.dumps({k:row.get(k) for k in ('worker','status','median_boundary_seconds','peak_reserved_bytes','total_memory_bytes')}))
PY
for log in /workspace/run057-control/preflight-*.log; do
    test ! -f "$log" || { echo "$log"; tail -n 2 "$log"; }
done
test ! -f /workspace/run057-control/pipeline.exit || cat /workspace/run057-control/pipeline.exit
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
