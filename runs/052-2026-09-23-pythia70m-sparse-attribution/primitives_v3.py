"""Exactly eight follow-ups selected from the first complete training screen."""
from functools import lru_cache
import torch
import legacy_ops
from support import RUN,sha
combine=legacy_ops.combine

def candidates(include_ablations=False):
    controls=[s for s in legacy_ops.candidates() if s['id'] in {
        'dense_native','dense_fused','dense_dot','c_m4_k32_n128','c_m8_k32_n128','c_m16_k32_n128'}]
    new=[{'id':f'f_vec_r{r}_v{v}','family':'f','strategy':'vector_warp','rows':r,'v':v,'load_v':v}
         for r in (4,8) for v in (4,8)]
    # v denotes the full output tile for the existing independent work oracle;
    # load_v denotes aligned per-thread loads. A compact CTA covers all512 outputs.
    new += [{'id':f'f_cta_v{v}','family':'f','strategy':'cta_compact','rows':1,'v':16,'load_v':v}
            for v in (2,4,8,16)]
    result=controls+new
    if include_ablations:
        result += [{**s,'id':s['id']+'_noskip','no_skip':True} for s in controls+new if s['family'] in ('c','f')]
    return result

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    path=RUN/'followup_sparse.cu'
    return load(name='r052_followup_'+sha(path)[:12],sources=[str(path)],
                extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)

class Linear:
    def __new__(cls,linear,threshold,spec,m=2048):
        if spec['family']!='f':return legacy_ops.Linear(linear,threshold,spec,m)
        return super().__new__(cls)
    def __init__(self,linear,threshold,spec,m=2048):
        self.spec=spec;self.w=linear.weight.t().contiguous();self.bias=linear.bias
        self.k,self.n=self.w.shape;self.m=m
        self.t=float(torch.tensor(threshold,dtype=torch.bfloat16)) if threshold is not None else -float('inf')
        self.out=torch.empty((m,self.n),device=self.w.device,dtype=torch.bfloat16)
        self.count=torch.empty((m,self.n//(32*spec['v'])),device=self.w.device,dtype=torch.int32)
        self.ext=extension();self.compiled=[]
    def __call__(self,x):
        s=self.spec;x=x.reshape(self.m,self.k)
        if not x.is_contiguous():raise ValueError('Contiguous operands required')
        self.ext.followup(x,self.w,self.bias,self.out,self.count,self.t,s['load_v'],s['rows'],s['strategy']=='cta_compact',s.get('no_skip',False))
        return self.out
