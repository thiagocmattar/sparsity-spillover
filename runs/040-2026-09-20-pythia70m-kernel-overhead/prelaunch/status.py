"""Read-only summary of this run's persistent phase and attempt records."""
from pathlib import Path
import json
import time

r = Path('/workspace/run040')
out = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
for name in ('setup-runtime.exit', 'setup-001.exit', 'pipeline-001.exit', 'operators-started',
             'smoke-started', 'scientific-started', 'diagnostic-complete'):
    p = r/'runtime'/name
    if p.exists(): out[name] = p.read_text().strip()
for phase in ('smoke', 'scientific', 'optimized'):
    p = r/'artifacts'/phase/'summary-001.json'
    if p.exists():
        rows = json.loads(p.read_text())
        out[phase] = {'finished': len(rows), 'qualified': sum(bool(x['qualified']) for x in rows),
                      'worker_seconds': sum(x['elapsed_seconds'] for x in rows), 'last': rows[-1]}
for name in ('operator-checks', 'control-checks'):
    p = r/'artifacts'/(name+'.json')
    if p.exists():
        data = json.loads(p.read_text())
        out[name] = {'status': data['status'], 'checks': len(data['checks'])}
statuses = sorted((r/'artifacts/attempts').glob('*/status.json'), key=lambda p: p.stat().st_mtime)
if statuses:
    p = statuses[-1]
    out['latest'] = {'attempt': p.parent.name, **json.loads(p.read_text())}
if not statuses:
    for log in ('setup-runtime.log', 'setup-001.log', 'operators-001.log'):
        p = r/'runtime'/log
        if p.exists(): out[log] = p.read_text(errors='replace').splitlines()[-5:]
print(json.dumps(out, indent=2))
