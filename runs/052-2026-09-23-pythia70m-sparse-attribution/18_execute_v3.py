"""One bounded sequential stage, persistent logs, absolute workload deadline."""
import argparse,datetime,subprocess,sys,time
from support import RUN,write,event,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','references','screen','training','final','diagnostics'),required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--deadline-utc',required=True);a=p.parse_args()
    deadline=datetime.datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))
    if deadline.tzinfo is None:raise ValueError('Timezone-aware UTC deadline required')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    jobs=[]
    def bench(cid,phase,rep,selection):
        jobs.append(['06_benchmark.py','--attempt',f'{a.attempt}-{cid}-r{rep}','--condition',cid,'--phase',phase,'--replicate',str(rep),'--selection',selection])
    if a.stage in ('references','final'):
        for rep in range(1,4):
            order=['d05','c24','d10','c25','c00']
            if rep%2==0:order.reverse()
            for cid in order:bench(cid,'reference' if cid.startswith('d') or a.stage=='references' else 'final',rep,'reference-policy.json' if a.stage=='references' or cid.startswith('d') else 'selection-final.json')
    elif a.stage=='smoke':
        for cid in ('d05','c24','d10','c25','c00'):bench(cid,'smoke',1,'reference-policy.json')
    elif a.stage=='screen':
        jobs=[['03_operator_checks.py','--attempt',a.attempt+'-operators'],
              ['04_screen.py','--attempt',a.attempt+'-screen','--operators',a.attempt+'-operators']]
    elif a.stage=='diagnostics':
        jobs=[['08_diagnostics.py','--attempt',a.attempt+'-'+cid,'--condition',cid,'--selection','selection-final.json'] for cid in ('c00','c24','c25')]
    else:
        for cid in ('c24','c25'):bench(cid,'smoke',0,'selection-training-v3.json')
        for cid in ('c24','c25'):bench(cid,'training',1,'selection-training-v3.json')
    start=time.monotonic();complete=[]
    try:
        for i,job in enumerate(jobs):
            seconds=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-read(RUN/'config.json')['retrieval_reserve_seconds']
            if seconds<=0:raise TimeoutError('Retrieval reserve reached before next worker')
            event(dest,'worker',index=i+1,total=len(jobs),command=job,elapsed_seconds=time.monotonic()-start)
            with (dest/f'{i+1:02d}.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u','17_followup.py',*job],cwd=RUN,stdout=log,stderr=subprocess.STDOUT)
                write(dest/'worker.json',{'pid':process.pid,'command':job})
                try:code=process.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise TimeoutError('Workload deadline reached; reserve retained')
            if code:raise RuntimeError(f'Worker failed with code {code}: {job}')
            complete.append(job);write(dest/'completed.json',complete)
        write(dest/'result.json',{'status':'complete','stage':a.stage,'completed':complete,'elapsed_seconds':time.monotonic()-start})
        event(dest,'complete',elapsed_seconds=time.monotonic()-start)
    except Exception as exc:
        write(dest/'result.json',{'status':'failed','stage':a.stage,'error':str(exc),'completed':complete});raise


if __name__=='__main__':main()
