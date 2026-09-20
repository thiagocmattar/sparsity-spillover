"""Native a/m and attention, with N512 h/z output tiles."""
from pathlib import Path
from io_utils import RUN, module
from controls import original_norms, substitute

HERE = Path(__file__).resolve().parent

def install(model):
    saved = original_norms(model)
    frozen = module('run040_opt003_frozen', RUN/'kernel/candidate.py')
    metadata = frozen.install(model, shortcut=False, round_p=False, skip=True, projection_skip=True)
    for mode in ('native-am', 'native-attention'):
        substitute(model, mode, saved)
    joint = module('run040_opt003_joint', HERE/'joint.py')
    for layer in model.gpt_neox.layers:
        layer._run026_joint = joint.Joint(layer._run026_joint)
    return {**metadata, 'identity': 'run040-opt003', 'new_optimization_search': True,
            'input_projections': 'Native PyTorch linear with existing fused gates',
            'attention': 'Native causal SDPA with existing gated Q/K/V operands',
            'output_projections': 'M8 N512, eight compute warps; same short-row policy, K16 masks, and accumulation order',
            'output_projection_counter_limits': 'Unchanged M8 padded-M16 potential and scalar-product accounting; N-grid resized only',
            'dispatch': 'Same component and tile policy for both checkpoints'}
