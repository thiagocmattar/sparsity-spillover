python3 - <<'PY'
from pathlib import Path
import json
root=Path('/workspace/sparsity-spillover/runs/041-2026-09-20-pythia14m-hz-h-only-ol1')
c=Path('/workspace/run041-control')
v=json.loads((root/'artifacts/verification.json').read_text())
print(json.dumps({'status':v['status'],'conditions':[{'kappa':r['condition']['gate_threshold'],'validation_loss':r['final_validation_loss'],'h_z_zero':r['selected_site_exact_zero_fractions']} for r in v['conditions']],'collection_exit':(c/'collection.exit').read_text() if (c/'collection.exit').exists() else None,'collection_log':(c/'collection.log').read_text()[-1500:],'archives':[{ 'path':p.name,'bytes':p.stat().st_size} for p in (root/'transfer').glob('*')]}))
PY
