"""Qualified parallel sparse h/z plus token-major dense attention output."""
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent

def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_opt064_parent',RUN/'candidates/opt063/candidate.py')
    metadata=base.install(model)
    attention=module('run042_opt064_attention',HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=attention.Attention(skip=False,shortcut=False)
    return {**metadata,'identity':'opt064','attention_output_layout':'T,H,D',
            'attention_activation_skipping':False,'attention_change':'Output layout only; unchanged dense attention arithmetic.'}
