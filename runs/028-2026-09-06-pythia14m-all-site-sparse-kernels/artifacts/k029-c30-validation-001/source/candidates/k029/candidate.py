"""Two-split KV native-order attention, with K021/K018 sparse projections."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import hashlib
import os
import torch
from common import RUN,module,read_json,verify_record,sha256

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    folder=Path(__file__).parent
    for row in read_json(RUN/'runtime/vendor/inventory.json')['files']:verify_record(row)
    digest=hashlib.sha256(''.join(sha256(f) for f in sorted(folder.iterdir()) if f.is_file()).encode()).hexdigest()[:12]
    major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k029_'+digest,sources=[str(folder/'kernel.cu')],
        extra_include_paths=[str(folder),str(RUN/'runtime/vendor/flash/csrc/flash_attn/src'),str(RUN/'runtime/vendor/cutlass/include')],
        extra_cuda_cflags=['-O3','-lineinfo','--expt-relaxed-constexpr','--expt-extended-lambda'],verbose=True)

class Attention:
    def __init__(self,skip=True,shortcut=True):self.skip=skip;self.shortcut=shortcut;self.shape=None
    def __call__(self,q,k,v,scale,*,count=False):
        if torch.is_grad_enabled() or q.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        if tuple(q.shape)!=(1,4,2048,32):raise ValueError('B1 H4 T2048 D32 only')
        if self.shape!=q.shape:
            self.out=torch.empty_like(q)
            self.lse=torch.empty((4,2048),device=q.device,dtype=torch.float32)
            self.lse_accum=torch.empty((2,4,2048),device=q.device,dtype=torch.float32)
            self.o_accum=torch.empty((2,4,2048,32),device=q.device,dtype=torch.float32)
            self.stats=torch.empty((4,32,2,4,4),device=q.device,dtype=torch.int64)
            self.prefix=torch.empty((2,4,1024,32),device=q.device,dtype=torch.float32)
            self.safe=torch.empty((2,4,32),device=q.device,dtype=torch.int32)
            self.prefix_stats=torch.empty((4,32,2,4,3),device=q.device,dtype=torch.int64)
            self.shape=q.shape
        extension().forward(q,k,v,self.out,self.lse,self.lse_accum,self.o_accum,self.stats,self.prefix,self.safe,self.prefix_stats,scale,self.skip,count,self.shortcut)
        return self.out

def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k029_base',RUN/'candidates/k020/candidate.py')
    projections=module('run028_k029_projections',RUN/'candidates/k021/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=Attention(skip,shortcut)
        layer.attention.forward=MethodType(base.attention_forward,layer.attention)
        layer._run026_joint.skip=projection_skip
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=projections.Projection(linear,skip=projection_skip)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    return {'sparse_operations':(['qkv_projection','mlp_w1','mlp_w2','attention_output_projection'] if projection_skip else [])+(['qk_scores','probability_value'] if skip else []),
        'dense_operations':['lm_head'],'attention':'k029 split KV plus exact-grid zero-query prefix','prefix_enabled':shortcut,
        'prefix_safety':'all K finite; all V zero or 0.5<=abs(V)<=64 within a split/head; entire 64-row query tile zero','attention_skipping_enabled':skip,
        'projection_skipping_enabled':projection_skip,'attention_skip_granularity':'16x8x16 MMA atoms with a fully zero A or B operand fragment',
        'attention_counter_unit':'issued/skipped MMA atoms; each atom is 2048 FMA-equivalent terms including padding',
        'precision':'UNFUSE_FMA, no fast-math, native two-split softmax and recombination'}
