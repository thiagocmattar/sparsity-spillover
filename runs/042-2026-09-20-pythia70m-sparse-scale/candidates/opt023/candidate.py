"""Sparse h/z plus dense attention query/key tile 64/128, 4 warps."""
from pathlib import Path
from io_utils import RUN, module
HERE = Path(__file__).resolve().parent

def install(model):
    base = module('run042_opt023_base', RUN/'base70/candidate.py')
    metadata = base.install(model)
    attention = module('run042_opt023_attention', HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention = attention.Attention(skip=False,shortcut=False)
    return {**metadata,'identity':'opt023','sparse_sites':['h','z'],
             'attention_activation_skipping':False,'attention_tile':[32,128],
             'attention_warps':2,'reversed_query_schedule':False,'attention_change':'Dense scheduling; separate from sparsity gain'}
