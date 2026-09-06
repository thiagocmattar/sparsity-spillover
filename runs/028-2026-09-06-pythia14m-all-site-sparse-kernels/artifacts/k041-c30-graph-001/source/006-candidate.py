"""Fused exact Q/K/V postprocessing, optionally composed with joint W2/Wo."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import hashlib
import os
import torch
from transformers.models.gpt_neox import modeling_gpt_neox
from sparsity_research.pythia import _optional_gate
from sparsity_research.sites import FixedSymmetricThreshold


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('kernel.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run026_k019_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12],
        sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class RopeGate:
    def __init__(self,attention,backend=None):
        self.heads=attention.config.num_attention_heads
        self.dimension=attention.head_size
        self.thresholds=[getattr(attention,f'{site}_gate').kappa for site in ['q_post','k_post','v']]
        self.backend=backend
        self.buffers=None

    def __call__(self,x,cos,sin):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16:
            raise ValueError('BF16 inference only')
        shape=(x.shape[0],self.heads,x.shape[1],self.dimension)
        if self.buffers is None or self.buffers[0].shape!=shape:
            self.buffers=tuple(torch.empty(shape,device=x.device,dtype=x.dtype) for _ in range(3))
        (self.backend or extension().forward)(x,cos.contiguous(),sin.contiguous(),*self.buffers,
                                            self.heads,self.dimension,*self.thresholds)
        return self.buffers


def attention_forward(self,hidden_states,attention_mask,layer_past=None,position_embeddings=None,**kwargs):
    if layer_past is not None or self.training: raise ValueError('Uncached inference only')
    input_shape=hidden_states.shape[:-1]
    raw=self.query_key_value(hidden_states)
    # Preserve inactive pre-RoPE taps for untimed instrumentation.
    qr,kr,_=raw.view(*input_shape,-1,3*self.head_size).transpose(1,2).chunk(3,dim=-1)
    self.q_pre_site(qr)
    self.k_pre_site(kr)
    q,k,v=self._run026_rope(raw,*position_embeddings)
    q,k,v=self.q_post_site(q),self.k_post_site(k),self.v_site(v)
    interface=modeling_gpt_neox.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,modeling_gpt_neox.eager_attention_forward)
    output,weights=interface(self,q,k,v,attention_mask,scaling=self.scaling,dropout=0.,**kwargs)
    output=output.reshape(*input_shape,-1).contiguous()
    output=self.z_site(_optional_gate(self,'z',output))
    return self.dense(output),weights


class Adapter:
    def __init__(self,model,joint=False,backend=None,joint_backend=None):
        if joint:
            from common import module,HERE
            module('k019_joint',HERE/'candidates/k018/candidate.py').Adapter(model,backend=joint_backend).install()
        for layer in model.gpt_neox.layers:
            attn=layer.attention
            if hasattr(attn,'q_pre_gate') or hasattr(attn,'k_pre_gate'):
                raise ValueError('Pre-RoPE gates are outside this candidate contract')
            if not all(isinstance(getattr(attn,f'{s}_gate'),FixedSymmetricThreshold) for s in ['q_post','k_post','v']):
                raise ValueError('Existing symmetric post-RoPE Q/K/V gates required')
            attn._run026_rope=RopeGate(attn,backend)
            attn.forward=MethodType(attention_forward,attn)
