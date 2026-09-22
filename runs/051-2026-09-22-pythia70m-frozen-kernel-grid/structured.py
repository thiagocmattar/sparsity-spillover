"""CUTLASS route preserves FP32 until overflow and bias have been combined."""
from functools import lru_cache
from support import RUN,BASE,sha

@lru_cache(None)
def extension():
 from torch.utils.cpp_extension import load
 from bootstrap import replay
 return load(name='r050_struct_'+sha(RUN/'structured.cu')[:12],sources=[str(RUN/'structured.cu')],
             extra_include_paths=[str(replay.R28/'runtime/vendor/cutlass/include')],
             extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)
