"""Native dense EFFICIENT_ATTENTION attention; unchanged gates and sparse h/z."""
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_opt062_parent',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    attention=module('run042_opt062_attention',HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=attention.Attention()
    return {**metadata,'identity':'opt062','attention_backend':'EFFICIENT_ATTENTION',
            'attention_activation_skipping':False,'attention':'Native dense SDPA backend; no changed operands, gates, mask or scaling.'}
