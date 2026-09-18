"""Read-only status snapshot for the single Run035 worker."""
import json
from pathlib import Path
import statistics
import time

root=Path('/workspace/run035')
for phase in ['smoke','qualify','scientific']:
    summaries=sorted((root/f'artifacts/{phase}').glob('summary-*.json'))
    if summaries:
        summary=summaries[-1]
        rows=json.loads(summary.read_text())
        total={'smoke':5,'qualify':6,'scientific':66}[phase]
        elapsed=[r['elapsed_seconds'] for r in rows if r['status']=='complete']
        print(json.dumps({'phase':phase,'completed':len(elapsed),'target':total,
                          'qualified':sum(r.get('qualified') is True for r in rows),
                          'rejected':[r['attempt'] for r in rows if r.get('qualified') is False],
                          'median_process_seconds':statistics.median(elapsed) if elapsed else None,
                          'estimated_remaining_seconds':(total-len(elapsed))*statistics.mean(elapsed) if elapsed else None,
                          'latest':rows[-1] if rows else None}))
paths=list((root/'artifacts/attempts').glob('*/status.json'))
if paths:
    latest=max(paths,key=lambda p:p.stat().st_mtime)
    print(json.dumps({'active_attempt':latest.parent.name,'age_seconds':time.time()-latest.stat().st_mtime,
                      'status':json.loads(latest.read_text())}))
else:print('No measurement attempt yet')
