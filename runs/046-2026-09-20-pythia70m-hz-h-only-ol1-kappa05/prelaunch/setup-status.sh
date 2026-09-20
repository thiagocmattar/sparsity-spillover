python3 - <<'PY'
from pathlib import Path
import json,time
c=Path('/workspace/run046-control')
r={'epoch':time.time(),'markers':[p.name for p in c.iterdir() if p.name.endswith(('ready','passed','verified','sealed','exit'))]}
for name in ['setup','cache','preflight-0','training-0','verification','collection']:
 p=c/(name+'.log')
 if p.exists():r[name]=p.read_text(errors='replace').splitlines()[-3:]
print(json.dumps(r,indent=2))
PY
