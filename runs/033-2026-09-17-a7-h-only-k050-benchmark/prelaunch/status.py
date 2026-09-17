"""Read-only status snapshot for the single Run033 worker."""
import json
from pathlib import Path
import statistics
import time

root=Path('/workspace/run033')
for phase in ['smoke','scientific']:
    summary=root/f'artifacts/{phase}/summary-001.json'
    if summary.exists():
        rows=json.loads(summary.read_text())
        total=2 if phase=='smoke' else 15
        elapsed=[r['elapsed_seconds'] for r in rows if r['status']=='complete']
        print(json.dumps({'phase':phase,'completed':len(elapsed),'target':total,
                          'median_process_seconds':statistics.median(elapsed) if elapsed else None,
                          'estimated_remaining_seconds':(total-len(elapsed))*statistics.median(elapsed) if elapsed else None,
                          'latest':rows[-1] if rows else None}))
paths=list((root/'artifacts/attempts').glob('*/status.json'))
if paths:
    latest=max(paths,key=lambda p:p.stat().st_mtime)
    print(json.dumps({'active_attempt':latest.parent.name,'age_seconds':time.time()-latest.stat().st_mtime,
                      'status':json.loads(latest.read_text())}))
else:print('No measurement attempt yet')
