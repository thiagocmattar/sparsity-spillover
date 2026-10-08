python3 - <<'PY'
import json,subprocess
from pathlib import Path
c=Path('/workspace/run066-control')
r=Path('/workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates')
status={'stage_markers':{n:(c/n).exists() for n in ['environment-ready','cache-ready','preflight-passed','completed']}}
status['pipeline_exit']=(c/'pipeline.exit').read_text().strip() if (c/'pipeline.exit').exists() else None
status['logs']={p.name:p.read_text(errors='replace').splitlines()[-8:] for p in sorted(c.glob('*.log'))}
status['preflights']=[]
for p in sorted((r/'prelaunch/attempts/002').glob('remote-preflight-*.json')):
    try:
        x=json.loads(p.read_text());status['preflights'].append({k:x.get(k) for k in ['worker','status','median_boundary_seconds','peak_reserved_bytes','total_memory_bytes','checkpoint_numerical_smoke']})
    except json.JSONDecodeError:pass
status['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
status['disk']=subprocess.run(['df','-h','/workspace'],capture_output=True,text=True).stdout.strip()
py=Path('/workspace/run066-venv/bin/python')
if (c/'environment-ready').exists() and (r/'artifacts/attempts').exists():
    result=subprocess.run([str(py),str(r/'04_monitor.py')],capture_output=True,text=True)
    status['training']=json.loads(result.stdout) if result.returncode==0 else {'monitor_error':result.stderr[-1000:]}
print(json.dumps(status,indent=2))
PY
