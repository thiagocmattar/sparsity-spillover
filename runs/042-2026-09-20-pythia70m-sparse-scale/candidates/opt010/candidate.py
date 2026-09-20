"""Sparse h/z plus dense attention query/key tile 64/256, 4 warps."""
from pathlib import Path
from io_utils import RUN, module
HERE = Path(__file__).resolve().parent

def install(model):
    base = module('run042_opt010_base', RUN/'base70/candidate.py')
    metadata = base.install(model)
    attention = module('run042_opt010_attention', HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention = attention.Attention(skip=False,shortcut=False)
    return {**metadata,'identity':'opt010','sparse_sites':['h','z'],
             'attention_activation_skipping':False,'attention_tile':[64,256],
             'attention_warps':4,'attention_change':'Dense scheduling; separate from sparsity gain'}
