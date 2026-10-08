python3 - <<'PY'
import json
from pathlib import Path
r=Path('/workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates')
c=Path('/workspace/run066-control')
s={'pipeline_exit':(c/'pipeline.exit').read_text().strip() if (c/'pipeline.exit').exists() else None,'completed_marker':(c/'completed').exists()}
v=r/'artifacts/verification.json'
s['verification']=json.loads(v.read_text()) if v.exists() else None
s['controls']=[]
for p in sorted((r/'artifacts/controls').glob('*/metrics.json')):
    x=json.loads(p.read_text());s['controls'].append({'id':p.parent.name,'validation':x['validation']})
s['seal_exit']=(r/'transfer/seal.exit').read_text().strip() if (r/'transfer/seal.exit').exists() else None
p=r/'transfer/training-receipt-001.json'
s['archive_receipt']=json.loads(p.read_text()) if p.exists() else None
s['pipeline_log_tail']=(c/'pipeline.log').read_text(errors='replace').splitlines()[-4:]
print(json.dumps(s,indent=2))
PY
