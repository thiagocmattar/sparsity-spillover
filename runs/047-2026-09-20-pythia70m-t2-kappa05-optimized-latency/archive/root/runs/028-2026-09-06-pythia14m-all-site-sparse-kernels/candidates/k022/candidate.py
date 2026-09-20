"""K022 all-site candidate: parallel sparse attention plus K021/K018 projections."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import os
import torch
from common import RUN,module,sha256

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('kernel.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k022_'+sha256(source)[:12],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)

base=module('run028_k022_base',RUN/'candidates/k020/candidate.py')

class Attention(base.Attention):
    def __call__(self,q,k,v,scale,*,count=False):
        if torch.is_grad_enabled():raise ValueError('Inference only')
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        if self.shape!=q.shape:
            b,h,t,d=q.shape
            if (b,d)!=(1,32) or not 0<t<=2048:raise ValueError('B1,D32,T<=2048')
            self.kt=torch.empty_like(q);self.values=torch.empty_like(q)
            self.indices=torch.empty(q.shape,device=q.device,dtype=torch.int32)
            self.counts=torch.empty(h*d,device=q.device,dtype=torch.int32)
            self.prefix=torch.empty(q.shape,device=q.device,dtype=torch.float32)
            self.out=torch.empty_like(q)
            self.stats=torch.empty((h,t,3),device=q.device,dtype=torch.int64)
            self.shape=q.shape
        extension().forward(q,k,v,self.kt,self.values,self.indices,self.counts,self.prefix,self.out,self.stats,scale,self.shortcut,self.round_p,count)
        return self.out

def install(model,shortcut=True,round_p=False):
    projections=module('run028_k022_projections',RUN/'candidates/k021/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=Attention(shortcut,round_p)
        layer.attention.forward=MethodType(base.attention_forward,layer.attention)
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=projections.Projection(linear)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    return {'sparse_operations':['qkv_projection','mlp_w1','mlp_w2','attention_output_projection','qk_scores','probability_value'],
            'dense_operations':['lm_head'],'attention':'k022','output_projections':'k018','input_projections':'k021'}
