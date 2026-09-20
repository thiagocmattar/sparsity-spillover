"""Exact gate/linear fusion; every graph replay recomputes all activations."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import hashlib
import os
import torch
from torch import nn
from sparsity_research.sites import FixedOneSidedThreshold


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('kernel.cu')
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(name=f'run026_k017_{identity}', sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo'], verbose=True)


class GatedLinear(nn.Module):
    def __init__(self, original, kappa, elements=4, warps=4, backend=None):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.kappa, self.elements, self.warps = kappa, elements, warps
        self.backend = backend
        self.weight_t = self.out = self.empty = None

    def forward(self, x):
        if self.training or torch.is_grad_enabled() or x.dtype != torch.bfloat16:
            raise ValueError('Inference-only BF16 operator')
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if self.weight_t is None:
            self.weight_t = self.weight.T.contiguous()
            self.empty = torch.empty(0, device=x.device, dtype=x.dtype)
        shape = (flat.shape[0], self.weight.shape[0])
        if self.out is None or self.out.shape != shape:
            self.out = torch.empty(shape, device=x.device, dtype=x.dtype)
        (self.backend or extension().linear)(flat, self.weight_t,
            self.bias if self.bias is not None else self.empty, self.out,
            self.kappa, self.elements, self.warps)
        return self.out.view(*x.shape[:-1], shape[-1])


def bypass(self, value):
    return value


class Adapter:
    def __init__(self, model, sites=('h', 'z'), elements=4, warps=4, backend=None):
        if not set(sites) <= {'a', 'm', 'h', 'z'} or not sites:
            raise ValueError('Invalid sites')
        self.entries = []
        for layer in model.gpt_neox.layers:
            mapping = {
                'a': (layer.attention, 'query_key_value', layer.a_gate),
                'm': (layer.mlp, 'dense_h_to_4h', layer.m_gate),
                'h': (layer.mlp, 'dense_4h_to_h', layer.mlp.act),
                'z': (layer.attention, 'dense', layer.attention.z_gate)}
            for site in sites:
                parent, name, gate = mapping[site]
                if not isinstance(gate, FixedOneSidedThreshold):
                    raise ValueError('Fusion requires the existing one-sided gate')
                original = getattr(parent, name)
                wrapped = GatedLinear(original, gate.kappa, elements, warps, backend).eval()
                self.entries.append((parent, name, original, wrapped, gate, gate.forward))

    def set_mode(self, sparse):
        for parent, name, original, wrapped, gate, forward in self.entries:
            setattr(parent, name, wrapped if sparse else original)
            gate.forward = MethodType(bypass, gate) if sparse else forward
