"""Integer-pooled occupancy for the actual gated operands; never tune on validation."""
import torch

class Structure:
 def __init__(self):self.data={}
 def add(self,key,x):
  x=x.reshape(-1,x.shape[-1]);m,k=x.shape;active=x!=0
  d=self.data.setdefault(key,{})
  def accum(name,v):
   v=v.detach().to(torch.float64 if v.is_floating_point() else torch.int64)
   if name not in d:d[name]=v.clone()
   else:d[name]+=v
  rows=active.sum(-1)
  accum('row_nnz_hist',torch.bincount(rows,minlength=k+1))
  accum('segment256_nnz_hist',torch.bincount(active.reshape(-1,256).sum(-1),minlength=257))
  accum('total',torch.tensor(x.numel(),device=x.device,dtype=torch.int64))
  accum('exact_zero',(x==0).sum());accum('finite',torch.isfinite(x).sum())
  for eps in (.001,.01):accum('near_zero_'+str(eps),(x.abs()<=eps).sum())
  xf=x.double();accum('sum',xf.sum());accum('sum_squares',(xf*xf).sum());accum('absolute_sum',xf.abs().sum())
  four=active.reshape(m,k//4,4).sum(-1)
  accum('four_feature_nnz_hist',torch.bincount(four.flatten(),minlength=5))
  for gm in (1,4,8,16,32):
   union=active.reshape(m//gm,gm,k).any(1);counts=union.sum(-1)
   accum(f'union{gm}_hist',torch.bincount(counts,minlength=k+1))
   for bk in (16,32,64,128,256):
    empty=~union.reshape(m//gm,k//bk,bk).any(-1)
    accum(f'empty_M{gm}_K{bk}',empty.sum())
   for bk in (32,64):accum(f'compact_padded_K_M{gm}_K{bk}',(((counts+bk-1)//bk)*bk).sum())
  for bm in (16,32,64):
   for bk in (32,64):
    bad=(four>2).reshape(m//bm,bm,k//bk,bk//4).any(3).any(1)
    accum(f'overflow24_M{bm}_K{bk}',bad.sum())
    accum(f'tiles24_M{bm}_K{bk}',torch.tensor(bad.numel(),device=x.device))
 def result(self):
  out={}
  for key,d in self.data.items():
   v={k:(x.cpu().tolist() if x.ndim else x.item()) for k,x in d.items()}
   v['rms']=(v['sum_squares']/v['finite'])**.5
   v['l2']=v['sum_squares']**.5
   out[key]=v
  return out

def gated(x,threshold):
 return x.masked_fill(x<threshold,0) if threshold is not None else x
