"""Inference-only K049 consumer with exact scalar/corner BF16 gates."""
from functools import lru_cache
import os
import torch
from io_utils import RUN, sha


def gate_spec(gate):
    from sparsity_research.sites import FixedOneSidedThreshold
    from tile_gate import CornerTileThreshold
    if isinstance(gate,FixedOneSidedThreshold):
        mode=1
    elif isinstance(gate,CornerTileThreshold) and (gate.tile_tokens,gate.tile_features)==(8,16) and gate.comparison=='signed':
        mode=2
    else:
        raise ValueError('Only the approved scalar or corner8x16 gates are supported')
    # Match TensorIterator's conversion of the wrapped scalar to the input dtype.
    return {'mode':mode,'threshold':float(torch.tensor(gate.kappa,dtype=torch.bfloat16)),
            'operator':'scalar' if mode==1 else 'corner8x16'}


def gated_value(value, mode, threshold):
    """Independent operand reconstruction for untimed counter verification."""
    if mode==1:
        return value.masked_fill(value<threshold,0)
    if mode==2:
        from tile_gate import CornerTileThreshold
        return CornerTileThreshold(threshold)(value)
    if mode==0:
        return value
    raise ValueError(mode)


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=RUN/'joint.cu'
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run062_joint_'+sha(source)[:12],sources=[str(source)],
                extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class Joint:
    def __init__(self,previous,h_gate,z_gate,*,skip):
        self.w2,self.wo=previous.w2,previous.wo
        self.wht,self.wzt=previous.wht,previous.wzt
        self.fast_weights=previous.fast_weights
        self.gh,self.th=h_gate['mode'],h_gate['threshold']
        self.gz,self.tz=z_gate['mode'],z_gate['threshold']
        self.skip,self.count=skip,False
        self.out=None

    def __call__(self,h,z,residual):
        if (torch.is_grad_enabled() or h.dtype!=torch.bfloat16 or h.ndim!=3 or z.ndim!=3
                or h.shape[:2]!=z.shape[:2] or residual.shape!=z.shape
                or h.shape[-1]!=512 or z.shape[-1]!=128 or h.shape[1]%8):
            raise ValueError('Complete sequence-local8-row blocks, BF16 inference only')
        original_shape=residual.shape
        h=h.reshape(-1,512).contiguous()
        z=z.reshape(-1,128).contiguous()
        residual=residual.reshape(-1,128).contiguous()
        if self.out is None or self.out.shape!=residual.shape:
            self.out=torch.empty_like(residual)
            self.stats=torch.empty((residual.shape[0]//8,8,6),device=residual.device,dtype=torch.int64)
        extension().forward(h,z,self.w2.weight,self.wo.weight,self.wht,self.wzt,
                            self.w2.bias,self.wo.bias,residual,self.out,self.stats,
                            self.th,self.tz,self.gh,self.gz,self.skip,self.fast_weights,self.count)
        return self.out.view(original_shape)
