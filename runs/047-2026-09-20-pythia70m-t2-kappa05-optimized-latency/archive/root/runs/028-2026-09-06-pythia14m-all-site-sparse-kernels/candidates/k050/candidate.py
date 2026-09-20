"""K049 matmuls with a shared-input, distinct-affine LayerNorm pair."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import math
import os
import torch
from common import RUN, module, sha256
from sparsity_research.sites import FixedOneSidedThreshold


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('norm.cu');major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}';os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k050_'+sha256(source)[:12],sources=[str(source)],
                extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


def gate_spec(gate):
    if gate is None:return False,0.
    if not isinstance(gate,FixedOneSidedThreshold):raise ValueError('Existing one-sided gate required')
    # TensorIterator casts a wrapped scalar to the BF16 tensor dtype.
    return True,float(torch.tensor(gate.kappa,dtype=torch.bfloat16))


class NormPair:
    def __init__(self,first,second,gate_a=None,gate_m=None):
        for norm in [first,second]:
            if not isinstance(norm,torch.nn.LayerNorm) or tuple(norm.normalized_shape)!=(128,):
                raise ValueError('Width128 LayerNorm required')
            if norm.weight is None or norm.bias is None:raise ValueError('Affine weight and bias required')
        if first.eps!=second.eps or not math.isfinite(first.eps) or first.eps<=0:
            raise ValueError('Equal positive finite epsilon required')
        self.first,self.second=first,second
        self.ga,self.ta=gate_spec(gate_a);self.gm,self.tm=gate_spec(gate_m)
        self.pending=None;self.out_a=None;self.out_m=None

    def start(self,x):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
            raise ValueError('Contiguous CUDA BF16 inference required')
        if x.shape[-1]!=128:raise ValueError('Width128 input required')
        if self.pending is not None:raise ValueError('Previous normalization pair not consumed')
        if self.out_a is None or self.out_a.shape!=x.shape:
            self.out_a=torch.empty_like(x);self.out_m=torch.empty_like(x)
        extension().forward(x,self.first.weight,self.first.bias,self.second.weight,self.second.bias,
            self.out_a,self.out_m,self.first.eps,self.ga,self.gm,self.ta,self.tm)
        self.pending=x
        return self.out_a

    def finish(self,x):
        if self.pending is not x:raise ValueError('Second normalization must consume the same current input')
        self.pending=None
        return self.out_m


def bind_pair(layer,pair):
    # Keep modules and their hooks. The existing gate is executed in norm.cu.
    layer.input_layernorm.forward=MethodType(lambda _self,x:pair.start(x),layer.input_layernorm)
    layer.post_attention_layernorm.forward=MethodType(lambda _self,x:pair.finish(x),layer.post_attention_layernorm)
    for name in ['a_gate','m_gate']:
        gate=getattr(layer,name,None)
        if gate is not None:gate.forward=MethodType(lambda _self,value:value,gate)
    layer._run028_norm_pair=pair


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    if not model.config.use_parallel_residual:raise ValueError('Parallel residual required')
    # Validate all pairs before any module mutation.
    pairs=[NormPair(layer.input_layernorm,layer.post_attention_layernorm,
                   getattr(layer,'a_gate',None),getattr(layer,'m_gate',None)) for layer in model.gpt_neox.layers]
    base=module('run028_k050_base',RUN/'candidates/k049/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer,pair in zip(model.gpt_neox.layers,pairs):bind_pair(layer,pair)
    metadata['normalization']='k050 shared-input width128 Welford pair, distinct affine, existing a/m BF16 gates fused'
    metadata['normalization_attribution']='Identical fusion in all skip controls; compare unfused K049 separately'
    return metadata
