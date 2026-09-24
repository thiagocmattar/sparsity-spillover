"""Segmented exact-grid sparse attention, with K033 tensor projections."""
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
    return load(name='run054_attention_'+digest,sources=[str(folder/'kernel.cu')],
        extra_include_paths=[str(folder),str(RUN/'runtime/vendor/flash/csrc/flash_attn/src'),str(RUN/'runtime/vendor/cutlass/include')],
        extra_cuda_cflags=['-O3','-lineinfo','--expt-relaxed-constexpr','--expt-extended-lambda','--ptxas-options=-v'],verbose=True)

class Attention:
    native = True  # Dense schedule; no sparse-attention counters.
    def __init__(self,skip=False,shortcut=False):self.skip=skip;self.shortcut=shortcut;self.shape=None
    def __call__(self,q,k,v,scale,*,count=False):
        if torch.is_grad_enabled() or q.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        if tuple(q.shape)!=(1,8,2048,32):raise ValueError('B1 H8 T2048 D32 only')
        if self.shape!=q.shape:
            self.out=torch.empty((1,2048,8,32),device=q.device,dtype=q.dtype).transpose(1,2)
            self.lse=torch.empty((8,2048),device=q.device,dtype=torch.float32)
            self.lse_accum=torch.empty((2,8,2048),device=q.device,dtype=torch.float32)
            self.o_accum=torch.empty((2,8,2048,32),device=q.device,dtype=torch.float32)
            self.stats=torch.empty((8,16,1,4,4),device=q.device,dtype=torch.int64)
            self.prefix=torch.empty((2,8,1024,32),device=q.device,dtype=torch.float32)
            self.safe=torch.empty((2,8,16,32),device=q.device,dtype=torch.int32)
            self.prefix_stats=torch.empty((8,16,1,4,3),device=q.device,dtype=torch.int64)
            self.shape=q.shape
        extension().forward(q,k,v,self.out,self.lse,self.lse_accum,self.o_accum,self.stats,self.prefix,self.safe,self.prefix_stats,scale,self.skip,count,self.shortcut)
        return self.out
