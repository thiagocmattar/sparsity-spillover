"""One read-only status snapshot of the owned Run042 process tree and records."""
import json
from pathlib import Path
import remote

COMMAND=r'''cd /workspace/run042 && python3 - <<'PY'
import json,time,subprocess
from pathlib import Path
def tail(path,n=3):
 p=Path(path)
 return p.read_text(errors='replace').splitlines()[-n:] if p.exists() else []
rows=[]
for p in sorted(Path('artifacts/attempts').glob('*/result.json')):
 d=json.loads(p.read_text())
 rows.append({'attempt':p.parent.name,'condition':d['condition'],'candidate':d['candidate'],
              'status':d['status'],'qualified':d.get('qualified'),'error':d.get('error'),
              'loss':d.get('loss',{}).get('candidate_graph'),
              'ms':d.get('timing',{}).get('candidate_graph',{}).get('median_host_ms')})
statuses=sorted(Path('artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
latest=[{'attempt':p.parent.name,**json.loads(p.read_text())} for p in statuses[-3:]]
checks=[]
for p in sorted(Path('artifacts/development').glob('operator-*.json')):
 d=json.loads(p.read_text());checks.append({'candidate':p.stem,'status':d['status'],'checks':len(d['checks'])})
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'remaining_minutes':(1789926479-time.time())/60,
 'setup_exit':tail('runtime/setup-001.exit'),'setup_log':tail('runtime/setup-001.log'),
 'pipeline_exit':tail('runtime/pipeline-001.exit'),'pipeline_log':tail('runtime/pipeline-001.log'),
 'operator_log':tail('runtime/operators-001.log'),'control_log':tail('runtime/controls-001.log'),
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip(),
 'completed':len(rows),'failed':[r for r in rows if not r['qualified']],
 'latest':latest,'operator_checks':checks,'recent_results':rows[-6:]}))
PY'''

def main():
 client=remote.connect()
 result=remote.execute(client,COMMAND,timeout=30)
 value=json.loads(result)
 print(json.dumps(value,indent=2))
 path=Path(__file__).with_name('status.jsonl')
 with path.open('a') as f:f.write(json.dumps(value)+'\n')
 client.close()

if __name__=='__main__':main()
