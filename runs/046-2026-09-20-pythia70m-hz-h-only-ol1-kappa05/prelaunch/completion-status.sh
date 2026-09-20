python3 - <<'PY'
from pathlib import Path
import json,time
r=Path('/workspace/sparsity-spillover/runs/046-2026-09-20-pythia70m-hz-h-only-ol1-kappa05')
c=Path('/workspace/run046-control')
v=r/'artifacts/verification.json'
out={'epoch':time.time(),'verified':(c/'training-verified').exists(),'sealed':(c/'training-sealed').exists(),'pipeline_exit':(c/'pipeline.exit').read_text() if (c/'pipeline.exit').exists() else None}
for name in ['verification','collection','pipeline-003']:
 p=c/(name+'.log')
 out[name]=p.read_text(errors='replace')[-1800:] if p.exists() else None
if v.exists():
 data=json.loads(v.read_text())
 out['status']=data['status']
 out['conditions']=[{'kappa':row['condition']['gate_threshold'],'validation_loss':row['final_validation_loss'],'R_model':row['R_model'],'h_z_zero':row['selected_site_exact_zero_fractions']} for row in data['conditions']]
out['archives']=[{'name':p.name,'bytes':p.stat().st_size} for p in (r/'transfer').glob('*') if p.is_file()]
print(json.dumps(out,indent=2))
PY
