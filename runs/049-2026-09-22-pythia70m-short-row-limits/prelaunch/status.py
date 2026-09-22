"""One bounded read-only status snapshot, called after the monitoring interval."""
import remote

client=remote.connect()
try:
    print(remote.execute(client,'''python3 - <<'PY'
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path('/workspace/run049');c=pathlib.Path('/workspace/run049-control')
print('utc',datetime.datetime.now(datetime.timezone.utc).isoformat())
print('inputs_verified',(r/'runtime/setup-complete').exists())
for name in ('environment.exit','pipeline.exit'):
 p=c/name
 if p.exists():print(name,p.read_text())
p=r/'artifacts/controller-status.json'
if p.exists():
 d=json.loads(p.read_text());print('controller',json.dumps(d))
 leaf=r/'artifacts'/(d.get('active','')+'.log')
 if leaf.is_file():print('active_log_tail',leaf.read_text(errors='replace')[-1200:])
attempts=sorted((r/'artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
if attempts:
 d=json.loads(attempts[-1].read_text());d.pop('timing',None)
 print('latest_attempt',attempts[-1].parent.name,json.dumps(d))
for name in ('environment.log','pipeline.log'):
 p=c/name
 if p.exists():print(name,p.read_text(errors='replace')[-700:])
print('disk_free_bytes',{p:shutil.disk_usage(p).free for p in ('/tmp','/workspace')})
print(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True))
PY''',timeout=60),flush=True)
finally:client.close()
