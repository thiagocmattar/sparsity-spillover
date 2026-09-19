"""Read-only monitoring of this Pod's detached setup and evaluation."""
import remote

COMMAND = """python3 - <<'PY'
import json,time,subprocess
from pathlib import Path
r=Path('/workspace/run038'); out={'epoch':time.time()}
for name in ['setup-001.exit','execution-001.exit']:
 p=r/'runtime'/name
 out[name]=p.read_text().strip() if p.exists() else None
p=r/'artifacts/cuda-controls.json'
if p.exists():
 d=json.loads(p.read_text());out['cuda_controls']={'status':d['status'],'cases':len(d['cases'])}
for phase in ['smoke','scientific']:
 p=r/'artifacts'/phase/'summary-001.json'
 if p.exists():
  try:
   rows=json.loads(p.read_text());out[phase]={'completed':len(rows),'last':rows[-1] if rows else None}
   if phase=='scientific' and rows:
    mean=sum(x['elapsed_seconds'] for x in rows)/len(rows)
    out[phase].update(mean_seconds=mean,remaining_seconds=(15-len(rows))*mean)
  except json.JSONDecodeError: pass
states=sorted((r/'artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
if states:
 p=states[-1];out['latest_attempt']=p.parent.name
 try: out['latest_status']=json.loads(p.read_text())
 except json.JSONDecodeError: pass
for name in ['deadline-guard.log','setup-001.log','execution-001.log']:
 p=r/'runtime'/name
 if p.exists():out[name]=p.read_text(errors='replace').splitlines()[-6:]
for p in sorted((r/'artifacts/infrastructure').glob('*/execution.*')):
 if p.suffix=='.exit' or 'scientific' not in out:
  out[str(p.relative_to(r))]=[s[:260] for s in p.read_text(errors='replace').splitlines()[-4:]]
p=r/'artifacts/infrastructure/003-local-runtime/relocation.log'
if p.exists():out['runtime_relocation']=p.read_text(errors='replace').splitlines()[-4:]
out['compiled_extensions']=len(list(Path('/opt/run038-extensions').glob('*/*.so')))
if states and out.get('latest_status',{}).get('stage') in ['capture','imports','failed']:
 p=states[-1];phase=p.parent.name.split('-')[0]
 log=r/'artifacts'/phase/(p.parent.name+'.log')
 if log.exists():out['latest_process_log']=[s[:260] for s in log.read_text(errors='replace').splitlines()[-4:]]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,clocks.sm','--format=csv,noheader'],text=True).strip()
out['active_compilers']=subprocess.run(['pgrep','-c','-f','[n]vcc|[c]icc|[c]c1plus'],capture_output=True,text=True).stdout.strip()
print(json.dumps(out))
PY"""

if __name__ == '__main__':
    c=remote.connect()
    try: print(remote.execute(c,COMMAND))
    finally: c.close()
