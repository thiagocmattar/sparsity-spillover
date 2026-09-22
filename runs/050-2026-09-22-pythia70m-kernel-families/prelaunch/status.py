"""Bounded read-only status; call at the agreed monitoring interval."""
import argparse,json
from transport import connect,execute
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
assert a.attempt.replace('-','').isalnum()
c=connect()
try:
 s=c.open_sftp()
 for suffix in ('status.json','result.json'):
  try:
   with s.file(f'/workspace/run050/artifacts/{a.attempt}/{suffix}') as f:d=json.loads(f.read())
   if suffix=='result.json':
    d={k:v for k,v in d.items() if k in ('status','error','eligible','elapsed_seconds','qualification','loss','loss_delta','children','implementations')}
   print(suffix,json.dumps(d))
  except FileNotFoundError:pass
 try:
  with s.file(f'/workspace/run050/artifacts/{a.attempt}/operators.json') as f:rows=json.loads(f.read())
  print('operators',json.dumps({'cells':len(rows),'passed':sum(r['pass'] for r in rows),'failures':[{'id':r['candidate']['id'],'k':r['k'],'error':r.get('error','')[-450:]} for r in rows if not r['pass']]}))
 except FileNotFoundError:pass
 if d.get('child'):
  try:
   with s.file(f"/workspace/run050/artifacts/{d['child']}/status.json") as f:print('child',f.read().decode())
  except FileNotFoundError:pass
 s.close()
 print(execute(c,f'tail -c 900 /workspace/run050-control/{a.attempt}.log; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader'))
finally:c.close()
