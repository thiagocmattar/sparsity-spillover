python3 - <<'PY'
from pathlib import Path
import json
p=Path('/workspace/sparsity-spillover/runs/041-2026-09-20-pythia14m-hz-h-only-ol1/inputs/random-initialization/model.safetensors')
c=Path('/workspace/run041-control')
print(json.dumps({'initializer_bytes_uploaded':p.stat().st_size if p.exists() else None,'pipeline_pid_exists':(c/'pipeline.pid').exists(),'pipeline_exit':(c/'pipeline.exit').read_text() if (c/'pipeline.exit').exists() else None}))
PY
