"""Run the frozen final comparison, agreed diagnostics, then counter probes."""
import argparse,subprocess,sys,time
from support import RUN,read,write,event

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--selection',default='selection-final.json');a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False);start=time.monotonic();finished=[]
 jobs=[('final-001','06_execute.py',['--phase','final'])]
 jobs += [('diagnostics-'+cid+'-001','07_diagnostics.py',['--condition',cid,'--phase','final']) for cid in ('c00','c24','c25')]
 jobs += [('hardware-001','09_hardware.py',[])]
 for child,script,extra in jobs:
  event(dest,'running',child=child,completed=len(finished),total=len(jobs),elapsed_seconds=time.monotonic()-start)
  with (dest/(child+'.log')).open('w') as f:
   proc=subprocess.run([sys.executable,'-u',str(RUN/script),'--attempt',child,'--selection',a.selection,*extra],stdout=f,stderr=subprocess.STDOUT)
  finished.append({'attempt':child,'exit_code':proc.returncode});write(dest/'children.json',finished)
  if proc.returncode:
   write(dest/'result.json',{'status':'failed','children':finished,'elapsed_seconds':time.monotonic()-start});return 1
 write(dest/'result.json',{'status':'complete','children':finished,'elapsed_seconds':time.monotonic()-start})
 event(dest,'complete',elapsed_seconds=time.monotonic()-start)
 return 0

if __name__=='__main__':raise SystemExit(main())
