python3 - <<'PY'
from pathlib import Path
import json,time
root=Path('/workspace/run043-latency');control=Path('/workspace/run043-control')
r={'epoch':time.time(),'pipeline_exit':(control/'pipeline.exit').read_text() if (control/'pipeline.exit').exists() else None,'completed':[],'active':[]}
for p in sorted((root/'artifacts/attempts').glob('scientific-*')):
 result=p/'result.json'
 if result.exists():
  row=json.loads(result.read_text());r['completed'].append({'attempt':p.name,'status':row['status'],'qualified':row.get('qualified'),'elapsed_seconds':row.get('elapsed_seconds'),'loss':row.get('loss'),'timing':row.get('timing')})
 elif (p/'status.json').exists():r['active'].append({'attempt':p.name,**json.loads((p/'status.json').read_text())})
r['completed_count']=len(r['completed']);r['qualified_count']=sum(x.get('qualified',False) for x in r['completed'])
r['mean_completed_process_seconds']=sum(x['elapsed_seconds'] for x in r['completed'])/max(1,len(r['completed']))
r['completed']=r['completed'][-2:]
if r['pipeline_exit'] is not None:
 for name in ['scientific','collection']:
  p=control/(name+'.log')
  if p.exists():r[name+'_log']=p.read_text(errors='replace').splitlines()[-3:]
print(json.dumps(r,indent=2))
PY
