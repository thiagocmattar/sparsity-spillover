"""One bounded read-only status snapshot, called after the monitoring interval."""
import remote

client=remote.connect()
try:
    print(remote.execute(client,'''python3 - <<'PY'
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path('/workspace/run049');c=pathlib.Path('/workspace/run049-control')
print('utc',datetime.datetime.now(datetime.timezone.utc).isoformat())
print('inputs_verified',(r/'runtime/setup-complete').exists())
bundle=pathlib.Path('/workspace/run049-input-001.tar.gz')
if bundle.exists():print('uploaded_bytes',bundle.stat().st_size,'target_bytes',788384619)
completed=[]
for p in sorted((r/'artifacts/attempts').glob('*/result.json'),key=lambda p:p.stat().st_mtime):
 d=json.loads(p.read_text())
 completed.append({'attempt':p.parent.name,'status':d['status'],'seconds':d.get('elapsed_seconds'),
                   'qualified':d.get('all_qualified'),'loss':d.get('loss'),
                   'timing_ms':{k:v.get('geomean_host_ms') for k,v in d.get('timing',{}).items()}})
if completed:print('completed_count',len(completed),'latest_completed',json.dumps(completed[-1:]))
for name in ('environment.exit','pipeline.exit','pipeline-002.exit'):
 p=c/name
 if p.exists():print(name,p.read_text())
p=r/'artifacts/controller-status.json'
if (r/'artifacts/recovery-001/controller-status.json').exists():p=r/'artifacts/recovery-001/controller-status.json'
if p.exists():
 d=json.loads(p.read_text());print('controller',json.dumps(d))
 leaf=r/'artifacts'/(d.get('active','')+'.log')
 if leaf.is_file():print('active_log_tail',leaf.read_text(errors='replace')[-700:])
attempts=sorted((r/'artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
if attempts:
 d=json.loads(attempts[-1].read_text());d.pop('timing',None)
 print('latest_attempt',attempts[-1].parent.name,json.dumps(d))
for name in ('environment.log','pipeline.log','pipeline-002.log'):
 p=c/name
 if name=='pipeline.log' and (c/'pipeline-002.log').exists():continue
 if name=='environment.log' and (c/'environment.exit').exists() and (c/'environment.exit').read_text().strip()=='0':continue
 if p.exists():print(name,p.read_text(errors='replace')[-700:])
print('disk_free_bytes',{p:shutil.disk_usage(p).free for p in ('/tmp','/workspace')})
print(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True))
PY''',timeout=60),flush=True)
finally:client.close()
