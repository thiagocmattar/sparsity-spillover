"""Complete the retained diagnostic inventory after a negative candidate search."""
import argparse,datetime,subprocess,sys,time
from support import RUN,read,write,event

p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--deadline-utc',required=True);a=p.parse_args()
deadline=datetime.datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))
assert deadline.tzinfo is not None
dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
started=time.monotonic();finished=[]
try:
    for i,cid in enumerate(('c00','c24','c25')):
        seconds=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-read(RUN/'config.json')['retrieval_reserve_seconds']
        if seconds<=0:raise TimeoutError('Retrieval reserve reached')
        job=['08_diagnostics.py','--attempt',a.attempt+'-'+cid,'--condition',cid,'--selection','diagnostic-reference-policy.json']
        event(dest,'worker',index=i+1,total=3,command=job,elapsed_seconds=time.monotonic()-started)
        with (dest/f'{i+1:02d}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-u',*job],cwd=RUN,stdout=log,stderr=subprocess.STDOUT)
            write(dest/'worker.json',{'pid':proc.pid,'command':job})
            try:code=proc.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise TimeoutError('Workload deadline reached')
        if code:raise RuntimeError(f'Diagnostic worker failed: {cid}, code{code}')
        finished.append(cid);write(dest/'completed.json',finished)
    write(dest/'result.json',{'status':'complete','conditions':finished,'elapsed_seconds':time.monotonic()-started})
    event(dest,'complete',elapsed_seconds=time.monotonic()-started)
except Exception as exc:
    write(dest/'result.json',{'status':'failed','error':str(exc),'completed':finished});raise
