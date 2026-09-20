python3 - <<'PY'
from pathlib import Path
import json,time
root=Path('/workspace/run043-latency');control=Path('/workspace/run043-control')
r={'epoch':time.time(),'pipeline_exit':(control/'pipeline.exit').read_text() if (control/'pipeline.exit').exists() else None,'attempts':[]}
for p in sorted((root/'artifacts/attempts').glob('*/status.json')):
 row=json.loads(p.read_text());r['attempts'].append({'attempt':p.parent.name,**row})
for name in ['setup','smoke','scientific','collection']:
 p=control/(name+'.log')
 if p.exists():r[name+'_log']=p.read_text(errors='replace').splitlines()[-3:]
print(json.dumps(r,indent=2))
PY
