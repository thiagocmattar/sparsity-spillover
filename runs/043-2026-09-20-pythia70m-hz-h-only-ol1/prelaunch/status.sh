python3 - <<'PY'
import json,os,time
from pathlib import Path
p=Path('/workspace/run043-control')
r={'epoch':time.time(),'pipeline_exit':(p/'pipeline.exit').read_text() if (p/'pipeline.exit').exists() else None,'markers':[x.name for x in p.iterdir() if x.name in ['cache-ready','preflight-passed','training-verified']]}
for x in sorted(p.glob('*.log')):
 with x.open('rb') as f:
  f.seek(max(0,x.stat().st_size-3500));s=f.read().decode(errors='replace')
 r[x.name]=s.splitlines()[-8:]
print(json.dumps(r,indent=2))
PY
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
