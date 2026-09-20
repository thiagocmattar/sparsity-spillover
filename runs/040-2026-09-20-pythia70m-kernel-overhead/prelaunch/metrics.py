"""Read completed attempts only; exploratory display is not the final reducer."""
from pathlib import Path
import json
import sys

r = Path('/workspace/run040')
prefix = sys.argv[1] if len(sys.argv) > 1 else 'scientific-'
rows = []
for p in sorted((r/'artifacts/attempts').glob(prefix+'*/result.json')):
    d = json.loads(p.read_text())
    rows.append({'attempt':p.parent.name, 'status':d['status'], 'qualified':d.get('qualified'),
                 'loss':d.get('loss'), 'median_host_ms':{k:v['median_host_ms'] for k,v in d.get('timing',{}).items()},
                 'seconds':d['elapsed_seconds']})
print(json.dumps(rows, indent=2))
