"""Sequential fresh processes and retained diagnostics under the lease deadline."""
import argparse,subprocess,sys,time
from support import RUN,read,write,event

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
 cfg=read(RUN/'config.json');start=time.monotonic();finished=[]
 if a.smoke:
  jobs=[(f'{a.attempt}-{cid}','02_benchmark.py',['--condition',cid,'--phase','smoke']) for cid in ('c22','c26','c07','c11')]
  jobs += [(f'{a.attempt}-diagnostics-c07','03_diagnostics.py',['--condition','c07','--phase','smoke'])]
 else:
  # Complete one replicate of every checkpoint before subsequent repeats.
  jobs=[(f'{a.attempt}-{cid}-r{rep}','02_benchmark.py',['--condition',cid,'--phase','final','--replicate',str(rep)]) for rep in (1,2,3) for cid in cfg['conditions']]
  jobs += [(f'{a.attempt}-diagnostics-{cid}','03_diagnostics.py',['--condition',cid,'--phase','final']) for cid in cfg['conditions']]
 for child,script,extra in jobs:
  elapsed=time.monotonic()-start
  event(dest,'running',child=child,completed=len(finished),total=len(jobs),elapsed_seconds=elapsed,remaining_seconds=None if not finished else elapsed/len(finished)*(len(jobs)-len(finished)))
  with (dest/(child+'.log')).open('w') as f:
   proc=subprocess.run([sys.executable,'-u',str(RUN/script),'--attempt',child,'--selection','selection-final.json',*extra],stdout=f,stderr=subprocess.STDOUT)
  result=read(RUN/'artifacts'/child/'result.json') if (RUN/'artifacts'/child/'result.json').exists() else {}
  finished.append({'attempt':child,'exit_code':proc.returncode,'qualified':result.get('qualification')});write(dest/'children.json',finished)
  if proc.returncode:
   write(dest/'result.json',{'status':'failed','children':finished,'elapsed_seconds':time.monotonic()-start});return 1
 write(dest/'result.json',{'status':'complete','children':finished,'elapsed_seconds':time.monotonic()-start})
 event(dest,'complete',completed=len(finished),total=len(jobs),elapsed_seconds=time.monotonic()-start)
 return 0

if __name__=='__main__':raise SystemExit(main())
