"""M8/N128 h/z with native a/m and attention; unchanged gates."""
from pathlib import Path
from io_utils import RUN, module
HERE = Path(__file__).resolve().parent

def install(model):
    base = module('run042_opt005_base', RUN/'base70/candidate.py')
    metadata = base.install(model)
    joint = module('run042_opt005_joint', HERE/'joint.py')
    for layer in model.gpt_neox.layers:
        layer._run026_joint = joint.Joint(layer._run026_joint)
    return {**metadata, 'identity':'opt005', 'tile_rows':8, 'tile_columns':128,
            'sparse_sites':['h','z'], 'attention_activation_skipping':False,
            'input_projection_skipping':False, 'unchanged_accumulation_order':True}
