"""Exact a/m and q/k/v gates composed with unchanged rounded dense producers."""
from functools import lru_cache
import os
import torch
from io_utils import RUN, sha, module, archived_run


def specification(gate):
    from sparsity_research.sites import FixedOneSidedThreshold, FixedSymmetricThreshold
    from tile_gate import CornerTileThreshold
    if isinstance(gate, FixedOneSidedThreshold):
        mode = 1
    elif isinstance(gate, FixedSymmetricThreshold):
        mode = 3
    elif isinstance(gate, CornerTileThreshold) and (gate.tile_tokens, gate.tile_features) == (8, 16):
        mode = 4 if gate.comparison == 'magnitude' else 2
    else:
        raise ValueError('Approved scalar or corner8x16 operator required')
    return mode, float(torch.tensor(gate.kappa, dtype=torch.bfloat16))


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    source = RUN / 'port_gates.cu'
    return load(name='run062_port_gates_' + sha(source)[:12], sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo'], verbose=True)


class Gates:
    def __init__(self, modules):
        specs = [specification(gate) for gate in modules]
        self.modes = [mode for mode, threshold in specs]
        self.thresholds = [threshold for mode, threshold in specs]
        self.outputs = None

    def __call__(self, inputs):
        if torch.is_grad_enabled():
            raise ValueError('Inference-only gates')
        inputs = list(inputs)
        if self.outputs is None or self.outputs[0].shape != inputs[0].shape:
            self.outputs = [torch.empty_like(value) for value in inputs]
        extension().forward(inputs, self.outputs, self.modes, self.thresholds)
        return self.outputs


class NormPair:
    def __init__(self, layer):
        frozen = module('run062_norm_producer', archived_run(28) / 'candidates/k050/candidate.py')
        self.raw = frozen.NormPair(layer.input_layernorm, layer.post_attention_layernorm)
        self.gates = Gates([layer.a_gate, layer.m_gate])
        self.pending = None

    def start(self, value):
        self.raw.start(value)
        self.pending = self.gates([self.raw.out_a, self.raw.out_m])
        return self.pending[0]

    def finish(self, value):
        self.raw.finish(value)
        result = self.pending[1]
        self.pending = None
        return result
