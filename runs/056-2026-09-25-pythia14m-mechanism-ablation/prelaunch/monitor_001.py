"""Read-only bounded snapshot for the approved Run056 worker."""
import json
import shlex
from ssh_001 import connect, RUN

SCRIPT = r'''
import json,subprocess,time
from pathlib import Path
p=Path('/workspace/run056-001')
r={'epoch':time.time()}
for name in ['setup.exit','worker.exit']:
 f=p/'runtime'/name
 if f.exists():r[name]=f.read_text().strip()
for phase in ['smoke','scientific']:
 f=p/'artifacts'/phase/'summary-001.json'
 if f.exists():
  rows=json.loads(f.read_text())
  r[phase]={'completed':len(rows),'qualified':sum(x['qualified'] for x in rows),'last':rows[-1]}
files=list((p/'artifacts/attempts').glob('*/status.json'))
if files:
 f=max(files,key=lambda f:f.stat().st_mtime)
 status=json.loads(f.read_text());status.pop('timing',None)
 r['latest']={'attempt':f.parent.name,**status}
 logs=list((p/'artifacts').glob('*/*'+f.parent.name+'.log'))
 if logs:r['child_tail']=logs[0].read_text(errors='replace')[-700:]
f=p/'runtime/worker.log'
if f.exists():r['worker_tail']=f.read_text(errors='replace')[-700:]
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip()
r['compiled_extensions']=len(list(Path('/opt/run056/extensions').glob('**/*.so')))
names=subprocess.check_output(['ps','-eo','comm'],text=True).splitlines()
r['compiler_processes']={name:names.count(name) for name in ['nvcc','cc1plus','c++','ptxas'] if name in names}
print(json.dumps(r))
'''

client=connect()
try:
 _,out,err=client.exec_command('python3 -c '+shlex.quote(SCRIPT),timeout=30)
 payload=out.read().decode();error=err.read().decode()
 if out.channel.recv_exit_status():raise RuntimeError(error)
 row=json.loads(payload)
 with (RUN/'prelaunch/monitor-001.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row))
finally:
 client.close()
