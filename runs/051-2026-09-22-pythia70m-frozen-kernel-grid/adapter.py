"""Independent site/layer choices with a zero-overhead Base fallback."""
import torch,triton
from primitives import Linear,combine,candidates

class Joint:
 native=True  # Compatibility: old diagnostics must not invent Run049 counters.
 def __init__(self,old,h,z):
  for name in ('w2','wo','gh','gz','th','tz'):setattr(self,name,getattr(old,name))
  self.h=Linear(self.w2,self.th if self.gh else None,h)
  self.z=Linear(self.wo,self.tz if self.gz else None,z)
  self.out=torch.empty((1,2048,512),device=self.w2.weight.device,dtype=torch.bfloat16)
 def __call__(self,h,z,residual):
  yh=self.h(h);yz=self.z(z)
  combine[(triton.cdiv(self.out.numel(),1024),)](yh,yz,residual,self.out,self.out.numel(),1024)
  return self.out

def install(model,policy):
 import bootstrap
 bootstrap.base_install.install(model,'native_hz')
 specs={s['id']:s for s in candidates(include_ablations=True)}
 choices=[]
 for index,layer in enumerate(model.gpt_neox.layers):
  old=layer._run026_joint
  if not old.gh and not old.gz:
   choices.append({'layer':index,'h':'native','z':'native','reason':'no active gates'})
   continue
  h=policy[f'h.{index}'];z=policy[f'z.{index}']
  layer._run026_joint=Joint(old,specs[h],specs[z])
  choices.append({'layer':index,'h':h,'z':z})
 return choices
