"""Native a/m and attention, retaining the frozen fused norm, RoPE, and sparse h/z."""
from io_utils import RUN, module
from controls import original_norms, substitute


def install(model):
    saved = original_norms(model)
    frozen = module('run040_opt001_frozen', RUN/'kernel/candidate.py')
    metadata = frozen.install(model, shortcut=False, round_p=False, skip=True, projection_skip=True)
    for mode in ('native-am', 'native-attention'):
        substitute(model, mode, saved)
    return {**metadata, 'identity': 'run040-opt001', 'new_optimization_search': True,
            'input_projections': 'Native PyTorch linear, with the existing fused a/m gates',
            'attention': 'Native causal SDPA, with the existing gated Q/K/V operands',
            'selection_split': 'first16 retained training-development blocks only',
            'dispatch': 'Same component policy for both checkpoints; no identity-dependent dispatch'}
