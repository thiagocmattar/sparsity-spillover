"""Compact read-only progress, ETC and task GPU-spend estimate."""
from pathlib import Path
import json
import time

r=Path('/workspace/run040')
now=time.time()
out={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),
     'estimated_gpu_usd':round((now-1789903519.103)*.99/3600,3)}
for phase,total in (('smoke',18),('scientific',54),('optimized',6)):
    p=r/'artifacts'/phase/'summary-001.json'
    if not p.exists():continue
    rows=json.loads(p.read_text())
    seconds=sum(x['elapsed_seconds'] for x in rows)
    out[phase]={'complete':len(rows),'target':total,'qualified':sum(bool(x['qualified']) for x in rows)}
    if phase!='smoke':
        out[phase].update(mean_process_seconds=round(seconds/len(rows),1),
                         estimated_remaining_minutes=round((total-len(rows))*seconds/len(rows)/60,1),
                         last={k:rows[-1].get(k) for k in ('condition','mode','replicate','loss')})
p=r/'artifacts/development/summary-001.json'
if p.exists():
    rows=json.loads(p.read_text())['rows'];out['development']={'records':len(rows),'last':rows[-1]}
paths=sorted((r/'artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
if paths:
    p=paths[-1];d=json.loads(p.read_text())
    out['active']={'attempt':p.parent.name,**{k:d[k] for k in ('stage','blocks','elapsed_seconds','remaining_seconds','input_tokens_per_second') if k in d}}
for name in ('pipeline-001.exit','development-001.exit','optimized-001.exit'):
    p=r/'runtime'/name
    if p.exists():out[name]=p.read_text().strip()
print(json.dumps(out,indent=2))
