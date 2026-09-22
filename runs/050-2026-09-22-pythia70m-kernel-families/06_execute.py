"""Sequential fresh processes for this run's matched checkpoint comparison."""
import argparse,subprocess,sys,time
from support import RUN,write,event

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--phase',choices=('smoke','training','final'),required=True)
 p.add_argument('--selection',required=True);a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False);start=time.monotonic();finished=[]
 jobs=[(cid,rep) for rep in range(1,4 if a.phase=='final' else 2) for cid in ('c00','c24','c25')]
 for cid,rep in jobs:
  attempt=f'{a.attempt}-{cid}-r{rep}'
  event(dest,'running',child=attempt,completed=len(finished),total=len(jobs),elapsed_seconds=time.monotonic()-start)
  with (dest/(attempt+'.log')).open('w') as f:
   result=subprocess.run([sys.executable,'-u',str(RUN/'05_benchmark.py'),'--attempt',attempt,'--phase',a.phase,'--selection',a.selection,'--condition',cid,'--replicate',str(rep)],stdout=f,stderr=subprocess.STDOUT)
  finished.append({'attempt':attempt,'exit_code':result.returncode})
  write(dest/'children.json',finished)
  if result.returncode:
   write(dest/'result.json',{'status':'failed','children':finished});return 1
 write(dest/'result.json',{'status':'complete','children':finished,'elapsed_seconds':time.monotonic()-start})
 event(dest,'complete',completed=len(finished),total=len(jobs),elapsed_seconds=time.monotonic()-start)
 return 0

if __name__=='__main__':raise SystemExit(main())
