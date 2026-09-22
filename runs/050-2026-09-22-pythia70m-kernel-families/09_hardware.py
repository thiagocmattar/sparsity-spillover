"""Four bounded, untimed hardware-counter probes, or a retained permission failure."""
import argparse,subprocess,sys,time
from pathlib import Path
from support import RUN,read,write,event,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--selection',required=True);a=p.parse_args()
 dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 selection=read(RUN/'provenance'/a.selection)
 ncu=Path('/opt/nvidia/nsight-compute/2025.1.1/ncu');results=[]
 if not ncu.exists():
  write(dest/'result.json',{'status':'unavailable','reason':'Nsight Compute executable absent'});return
 metrics='dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,launch__registers_per_thread'
 primary=next(iter(selection['policies'].values()),selection['dense'])
 for cid in ('c24','c25'):
  for label,policy in [('dense',selection['dense']),('sparse',primary)]:
   candidate=policy['h.1'];stem=f'{cid}-{label}'
   command=[str(ncu),'--csv','--page','raw','--profile-from-start','off','--target-processes','all','--cache-control','none','--clock-control','none','--metrics',metrics,'--log-file',str(dest/(stem+'.csv')),sys.executable,str(RUN/'08_counter_probe.py'),'--condition',cid,'--candidate',candidate,'--site','h.1']
   event(dest,'hardware_counter',condition=cid,candidate=candidate)
   with (dest/(stem+'.log')).open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240)
   text='\n'.join(p.read_text(errors='replace') for p in (dest/(stem+'.log'),dest/(stem+'.csv')) if p.exists())
   row={'condition':cid,'candidate':candidate,'training_block':0,'site':'h.1','exit_code':result.returncode,'command':command}
   results.append(row);write(dest/'attempts.json',results)
   if 'ERR_NVGPUCTRPERM' in text or 'permission' in text.lower():
    write(dest/'result.json',{'status':'unavailable','reason':'GPU performance-counter permission denied','attempts':results,'elapsed_seconds':time.monotonic()-start});return
 write(dest/'result.json',{'status':'complete' if all(r['exit_code']==0 for r in results) else 'failed','attempts':results,'selection_sha256':sha(RUN/'provenance'/a.selection),'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
