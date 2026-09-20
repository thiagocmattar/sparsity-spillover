"""One bounded read-only status snapshot of the owned Run045 Pod."""
import remote

client = remote.connect()
try:
    print(remote.execute(client, '''python3 - <<'PY'
import datetime, json, pathlib, statistics, subprocess, time
r=pathlib.Path('/workspace/run045');c=pathlib.Path('/workspace/run045-control')
print('utc',datetime.datetime.now(datetime.timezone.utc).isoformat())
print('inputs_verified',(r/'runtime/setup-complete').exists())
for p in [c/'environment.exit', c/'pipeline.exit', c/'tail.exit', c/'local-runtime-cache.exit',c/'local-archive-cache.exit']:
 if p.exists():print(p.name,p.read_text()[-5000:])
p=r/'artifacts/progress.json'
if p.exists():
 d=json.loads(p.read_text());latest=d.pop('latest');d['latest']={k:latest.get(k) for k in ('condition','replicate','seconds','qualified','status','loss','error')};print('progress',json.dumps(d))
summary=r/'artifacts/final/summary-tail001.json'
if not summary.exists():summary=r/'artifacts/final/summary-001.json'
order=r/'artifacts/final/order-001.json'
if summary.exists() and order.exists():
 rows=json.loads(summary.read_text()); jobs=json.loads(order.read_text())
 warm=[x for x in rows[6:] if x['seconds']<180][-12:]
 if len(warm)>=3:
  typical={rep:statistics.median([x['seconds'] for x in warm if (x['replicate']==1)==rep] or [x['seconds'] for x in warm]) for rep in (True,False)}
  remaining=jobs[len(rows):]; seconds=sum(typical[rep==1] for _,rep in remaining)
  print('warm_forecast',json.dumps({'seconds_including_active':seconds,'median_seconds_by_diagnostics':typical,'completion_utc':datetime.datetime.fromtimestamp(time.time()+seconds,datetime.timezone.utc).isoformat(),'qualified_completed':sum(x['qualified'] for x in rows)}))
attempts=sorted((r/'artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
if attempts:
 d=json.loads(attempts[-1].read_text());d.pop('timing',None)
 print('latest_attempt',attempts[-1].parent.name,json.dumps(d))
if attempts:
 p=r/'runtime'/(attempts[-1].parent.name+'.log')
 if p.exists():
  lines=p.read_text(errors='replace').splitlines()
  print('leaf_log_tail','\\n'.join(line[-250:] for line in lines[-2:]))
print(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True))
PY''', timeout=60), flush=True)
finally:
    client.close()
