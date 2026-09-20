import remote
c=remote.connect()
command=r"""cd /workspace/run042 && python3 - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('artifacts/development').glob('operator-stress-audit-*.json')):
 d=json.loads(p.read_text());rows=d['checks']
 print({'candidate':d['candidate']['candidate'],'status':d['status'],'checks':len(rows),'bitwise_frozen':d.get('all_bitwise_frozen'),'frozen_bounds':d.get('all_frozen_bounds'),'native_bounds':d.get('all_native_bounds'),'counts_match':all(r['counts']==r['oracle'] for r in rows),'failed_native_cases':[(r['case'],r['gate_h'],r['gate_z'],r['skip'],r['bitwise_equal_frozen'],r['frozen_passes_native_bound']) for r in rows if not r['pass']]})
PY"""
print(remote.execute(c,command));c.close()
