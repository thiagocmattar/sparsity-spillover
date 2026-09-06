"""K020 sparse QK/PV prototype. All dynamic preprocessing belongs to forward."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from common import sha256


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('kernel.cu')
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='run028_k020_'+sha256(source)[:12], sources=[str(source)],
                extra_cuda_cflags=['-O3','-lineinfo'], verbose=True)


class Attention:
    def __init__(self, shortcut=True, round_p=False):
        self.shortcut, self.round_p = shortcut, round_p
        self.shape = None

    def __call__(self, q, k, v, scale, *, count=False):
        if torch.is_grad_enabled():
            raise ValueError('Inference only')
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if self.shape != q.shape:
            b,h,t,d=q.shape
            if (b,d)!=(1,32) or not 0<t<=2048:
                raise ValueError('K020 supports B1,D32,T<=2048')
            self.kt=torch.empty_like(q)
            self.values=torch.empty_like(q)
            self.indices=torch.empty(q.shape,device=q.device,dtype=torch.int32)
            self.counts=torch.empty(h*d,device=q.device,dtype=torch.int32)
            self.prefix=torch.empty(q.shape,device=q.device,dtype=torch.float32)
            self.out=torch.empty_like(q)
            self.stats=torch.empty((h,t,3),device=q.device,dtype=torch.int64)
            self.shape=q.shape
        extension().forward(q,k,v,self.kt,self.values,self.indices,self.counts,
                            self.prefix,self.out,self.stats,scale,self.shortcut,self.round_p,count)
        return self.out


def install(model, shortcut=True, round_p=False):
    from types import MethodType
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=Attention(shortcut,round_p)
        layer.attention.forward=MethodType(attention_forward,layer.attention)


def attention_forward(self,hidden_states,attention_mask,layer_past=None,position_embeddings=None,**kwargs):
    # Preserve the frozen K019 preprocessing and canonical SDPA mask selection;
    # only the QK/softmax/PV component is replaced. No registry/global mutation.
    from sparsity_research.pythia import _optional_gate
    if layer_past is not None or self.training or attention_mask is not None:
        raise ValueError('Only unmasked ordinary uncached causal inference')
    shape=hidden_states.shape[:-1]
    raw=self.query_key_value(hidden_states)
    qr,kr,_=raw.view(*shape,-1,3*self.head_size).transpose(1,2).chunk(3,dim=-1)
    self.q_pre_site(qr);self.k_pre_site(kr)
    q,k,v=self._run026_rope(raw,*position_embeddings)
    q,k,v=self.q_post_site(q),self.k_post_site(k),self.v_site(v)
    output=self._run028_attention(q,k,v,self.scaling).transpose(1,2)
    output=output.reshape(*shape,-1).contiguous()
    output=self.z_site(_optional_gate(self,'z',output))
    return self.dense(output),None
