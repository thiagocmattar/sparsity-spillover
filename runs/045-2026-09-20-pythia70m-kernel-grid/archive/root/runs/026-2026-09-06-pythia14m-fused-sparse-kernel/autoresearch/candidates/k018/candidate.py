"""Fuse both sparse output branches and the two rounded residual additions."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import hashlib
import os
import torch
from sparsity_research.sites import FixedOneSidedThreshold


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('kernel.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run026_k018_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12],
                sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


def identity(self,x): return x


class Joint:
    def __init__(self,layer,backend=None):
        self.w2,self.wo=layer.mlp.dense_4h_to_h,layer.attention.dense
        self.th,self.tz=layer.mlp.act.kappa,layer.attention.z_gate.kappa
        self.backend=backend
        self.wh=self.wz=self.out=None

    def __call__(self,h,z,residual):
        if torch.is_grad_enabled() or h.dtype!=torch.bfloat16:
            raise ValueError('Inference BF16 only')
        h=h.reshape(-1,h.shape[-1]).contiguous()
        z=z.reshape(-1,z.shape[-1]).contiguous()
        r=residual.reshape(-1,residual.shape[-1]).contiguous()
        if self.wh is None:
            self.wh=self.w2.weight.T.contiguous()
            self.wz=self.wo.weight.T.contiguous()
        if self.out is None or self.out.shape!=r.shape:
            self.out=torch.empty_like(r)
        (self.backend or extension().forward)(h,z,self.wh,self.wz,self.w2.bias,
                self.wo.bias,r,self.out,self.th,self.tz)
        return self.out.view_as(residual)


def layer_forward(self,hidden_states,attention_mask=None,position_ids=None,
                  use_cache=False,layer_past=None,position_embeddings=None,**kwargs):
    if use_cache or layer_past is not None or self.training:
        raise ValueError('K018 requires uncached inference')
    z,_=self.attention(self.input_layernorm(hidden_states),attention_mask=attention_mask,
        position_ids=position_ids,layer_past=None,use_cache=False,
        position_embeddings=position_embeddings,**kwargs)
    h=self.mlp(self.post_attention_layernorm(hidden_states))
    return self._run026_joint(h,z,hidden_states)


class Adapter:
    def __init__(self,model,backend=None):
        self.layers=list(model.gpt_neox.layers)
        for layer in self.layers:
            if not layer.use_parallel_residual or not isinstance(layer.mlp.act,FixedOneSidedThreshold) or not isinstance(layer.attention.z_gate,FixedOneSidedThreshold):
                raise ValueError('Parallel residual and one-sided h/z gates required')
            if layer.mlp.dense_4h_to_h.bias is None or layer.attention.dense.bias is None:
                raise ValueError('Biases required')
            layer._run026_joint=Joint(layer,backend)

    def install(self):
        for layer in self.layers:
            layer.mlp.dense_4h_to_h=torch.nn.Identity()
            layer.attention.dense=torch.nn.Identity()
            layer.mlp.act.forward=MethodType(identity,layer.mlp.act)
            layer.attention.z_gate.forward=MethodType(identity,layer.attention.z_gate)
            layer.forward=MethodType(layer_forward,layer)
