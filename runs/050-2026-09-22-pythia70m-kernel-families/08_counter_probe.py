"""One real training operand, isolated for untimed Nsight hardware counters."""
import argparse
from support import RUN,read,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',default='c25');p.add_argument('--candidate',default='c_m4_k32');p.add_argument('--site',default='h.1');a=p.parse_args()
 import bootstrap
 bootstrap.setup()
 import torch,numpy as np
 from primitives import Linear,candidates
 manifest=read(RUN/'provenance/inputs.json')['training'];path=RUN/manifest['path'];assert sha(path)==manifest['sha256']
 ids=torch.tensor(np.fromfile(path,dtype=np.int32,count=2048).copy(),device='cuda',dtype=torch.long)[None]
 site,index=a.site.split('.');captured={}
 with torch.inference_mode():
  model=bootstrap.model(a.condition);layer=model.gpt_neox.layers[int(index)];old=layer._run026_joint
  class Observe:
   def __call__(self,h,z,r):captured['x']=h if site=='h' else z;return old(h,z,r)
  layer._run026_joint=Observe();bootstrap.replay.scaffold.forward(model,ids)
  op=Linear(old.w2 if site=='h' else old.wo,old.th if site=='h' else old.tz,next(s for s in candidates() if s['id']==a.candidate))
  x=captured['x'].reshape(2048,-1)
  for _ in range(3):op(x)
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
  op(x);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
  print({'condition':a.condition,'candidate':a.candidate,'site':a.site,'training_block':0,'input_shape':list(x.shape),'purpose':'instrumented hardware counters; not latency evidence'})

if __name__=='__main__':main()
