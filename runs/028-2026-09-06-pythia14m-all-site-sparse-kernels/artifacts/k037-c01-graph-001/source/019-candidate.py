"""All-six-site prototype: K020 attention, K018 h/z, new sparse a/m linears."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import os
import torch
from common import RUN,module,sha256


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('projection.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k021_'+sha256(source)[:12],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class Projection:
    def __init__(self,linear,skip=True):
        self.linear,self.skip=linear,skip
        self.weight=None;self.output=None

    def __call__(self,x):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        shape=x.shape[:-1]
        x=x.reshape(-1,128).contiguous()
        if self.weight is None:self.weight=self.linear.weight.T.contiguous()
        n=self.weight.shape[1]
        if self.output is None or self.output.shape!=(x.shape[0],n):
            self.output=torch.empty((x.shape[0],n),device=x.device,dtype=x.dtype)
        extension().forward(x,self.weight,self.linear.bias,self.output,self.skip)
        return self.output.view(*shape,n)


def install(model,shortcut=True,round_p=False):
    attention=module('run028_k021_attention',RUN/'candidates/k020/candidate.py')
    attention.install(model,shortcut,round_p)
    for layer in model.gpt_neox.layers:
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=Projection(linear)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    return {'sparse_operations':['qkv_projection','mlp_w1','mlp_w2','attention_output_projection','qk_scores','probability_value'],
            'dense_operations':['lm_head'],'attention':'k020','output_projections':'k018','input_projections':'k021'}
