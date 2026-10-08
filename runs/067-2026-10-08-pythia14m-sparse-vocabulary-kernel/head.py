"""FP32 corner certificates precede FP16 conversion; all vocabulary scores remain."""
from functools import lru_cache
import math
import os
import torch
from io_utils import RUN, record


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source = RUN/'head.cu'
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='run067_head_'+record(source)['sha256'][:12], sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo'], verbose=True)


def validate_gate(spec):
    if (not isinstance(spec, dict) or set(spec) != {'operator','kappa','tile_tokens','tile_features'}
            or spec['operator'] != 'tile_corner_threshold'
            or (spec['tile_tokens'],spec['tile_features']) != (8,16)
            or isinstance(spec['kappa'], bool) or not isinstance(spec['kappa'], (int,float))
            or not math.isfinite(spec['kappa']) or spec['kappa'] < 0):
        raise ValueError('An explicit unchanged corner8x16 gate is required')
    return float(torch.tensor(spec['kappa'], dtype=torch.float32))


def pack_weight(weight):
    """Untimed exact permutation: each MMA weight-fragment load becomes contiguous."""
    vocabulary, width = weight.shape
    if width!=128 or weight.dtype!=torch.float16:
        raise ValueError('FP16 V/D128 weights required')
    padded=torch.nn.functional.pad(weight,(0,0,0,(-vocabulary)%8))
    return padded.reshape(-1,8,8,2,8).permute(2,0,3,1,4).contiguous().reshape(8,-1,128)


class DenseHead(torch.nn.Module):
    """cuBLAS reference with the same untimed, static weight cast as the custom head."""
    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        started = __import__('time').monotonic()
        self.packed_weight = self.weight.detach().to(torch.float16).contiguous()
        torch.cuda.synchronize() if self.weight.is_cuda else None
        self.preparation_seconds = __import__('time').monotonic()-started

    def forward(self, value):
        return torch.nn.functional.linear(value.to(torch.float16), self.packed_weight)


class Head(torch.nn.Module):
    def __init__(self, linear, spec, *, skip, tile_tokens=16, tile_vocabulary=256):
        super().__init__()
        if linear.bias is not None or tuple(linear.weight.shape) != (50304,128):
            raise ValueError('The bias-free 14M full-vocabulary head is required')
        if tile_tokens not in (8,16) or tile_vocabulary not in (128,256,512):
            raise ValueError('Unsupported consumer tile')
        self.weight = linear.weight  # original parameter, original state_dict name
        self.threshold = validate_gate(spec)
        self.skip, self.tile_tokens, self.tile_vocabulary = skip, tile_tokens, tile_vocabulary
        started = __import__('time').monotonic()
        self.packed_weight = self.weight.detach().to(torch.float16).contiguous()
        self.mma_weight = pack_weight(self.packed_weight)
        torch.cuda.synchronize() if self.weight.is_cuda else None
        self.preparation_seconds = __import__('time').monotonic()-started
        self.x_half = self.support = self.out = self.stats = None

    def forward(self, value):
        if (torch.is_grad_enabled() or value.dtype != torch.float32 or not value.is_cuda
                or value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 128
                or value.shape[1] % self.tile_tokens or not value.is_contiguous()):
            raise ValueError('Contiguous FP32 B1/T-aligned/D128 uncached inference required')
        rows = value.shape[1]
        if self.out is None or self.out.shape[0] != rows:
            self.x_half = torch.empty((rows,128), device=value.device, dtype=torch.float16)
            self.support = torch.empty(rows//8, device=value.device, dtype=torch.int32)
            self.out = torch.empty((rows,50304), device=value.device, dtype=torch.float16)
            columns = (50304+self.tile_vocabulary-1)//self.tile_vocabulary
            self.stats = torch.empty((rows//self.tile_tokens,columns,2), device=value.device, dtype=torch.int64)
        extension().prepare(value.reshape(rows,128), self.x_half, self.support, self.threshold)
        return self.project().view(1,rows,50304)

    def project(self, *, count=False):
        extension().project(self.x_half, self.support, self.mma_weight, self.out,
                            self.stats, self.skip, count, self.tile_tokens, self.tile_vocabulary)
        return self.out


def install(model, mode, config):
    if mode not in ('dense','custom_no_skip','custom_sparse'):
        raise ValueError(mode)
    spec = model.config.head_input_gate
    validate_gate(spec)
    if mode == 'dense':
        model.embed_out = DenseHead(model.embed_out)
    else:
        model.embed_out = Head(model.embed_out, spec, skip=mode=='custom_sparse',
                              tile_tokens=config['kernel_tile_tokens'],
                              tile_vocabulary=config['kernel_tile_vocabulary'])
    return {'mode':mode, 'gate':spec, 'gate_materialized':True,
            'parameters':'unchanged FP32', 'projection':'FP16 inputs and weights, FP32 accumulation, FP16 logits',
            'static_weight_preparation':'identical FP16 cast for dense and custom, outside timing; qualified vs unmodified eager autocast',
            'weight_preparation_seconds':model.embed_out.preparation_seconds,
            'fp16_weight_bytes':model.embed_out.packed_weight.numel()*2,
            'additional_mma_weight_bytes':0 if mode=='dense' else model.embed_out.mma_weight.numel()*2,
            'certificate':'native post-gate FP32 corner > native FP32 threshold before cast',
            'consumer_tile':[config['kernel_tile_tokens'],config['kernel_tile_vocabulary'],16]}
