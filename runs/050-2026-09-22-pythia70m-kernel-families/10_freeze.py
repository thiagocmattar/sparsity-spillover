"""Freeze only after all training qualifications; no validation-based retuning."""
import argparse,time,math
from support import RUN,read,write,sha

KERNEL_SOURCES=('config.json','bootstrap.py','primitives.py','packed.cu','structured.cu','structured.py','adapter.py','structure.py','work_counters.py','05_benchmark.py','06_execute.py','07_diagnostics.py','support.py')

def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--training',required=True);p.add_argument('--output',default='selection-final.json');a=p.parse_args()
 target=RUN/'provenance'/a.output;assert not target.exists(),'Final selection is immutable'
 selection=read(RUN/'provenance'/a.selection);results={};proof=[]
 for cid in ('c00','c24','c25'):
  path=RUN/'artifacts'/f'{a.training}-{cid}-r1'/'result.json';r=read(path)
  assert r['status']=='complete' and r['validation_blocks']==128
  assert r['arguments']['phase']=='training' and r['arguments']['selection']==a.selection
  assert r['selection']['sha256']==sha(RUN/'provenance'/a.selection)
  for name in ('bootstrap.py','primitives.py','packed.cu','structured.cu','structured.py','adapter.py','support.py'):
   assert r['source_hashes'][name]==sha(RUN/name),('Untested kernel change',name)
  results[cid]=r;proof.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
 modes=['dense_fused','dense_policy',*selection['policies'],*selection.get('ablations',{})]
 qualified=[m for m in modes if all(r['qualification'][m+'_graph'] for r in results.values())]
 assert 'dense_policy' in qualified,'Shared dense policy failed training; do not proceed'
 sparse=[m for m in selection['policies'] if m in qualified]
 primary=min(sparse,key=lambda m:max(results[c]['timing'][m+'_graph']['geomean_host_ms']/results[c]['timing']['dense_policy_graph']['geomean_host_ms'] for c in ('c24','c25'))) if sparse else 'dense_policy'
 selection.update(qualified_modes=qualified,primary=primary,training_qualification=proof,
                  frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  numerical_bounds=read(RUN/'config.json')['numerical_bounds'],
                  kernel_source_hashes={name:sha(RUN/name) for name in KERNEL_SOURCES},
                  excluded_modes={m:'failed fixed training numerical gate' for m in modes if m not in qualified})
 write(target,selection);print({'primary':primary,'qualified_modes':qualified,'path':str(target),'sha256':sha(target)})

if __name__=='__main__':main()
