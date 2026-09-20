"""Measured h/z, attention and full-vocabulary components; gates unchanged."""
from types import MethodType
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent

def install(model):
    spec=read(HERE/'spec.json')
    for record in spec['parent_manifests'].values():
        manifest=read(verify(record))
        for item in manifest['files']:verify(item)
    parents=spec['parents']
    attention=module('run042_composed_attention',RUN/'candidates'/parents['attention']/'candidate.py')
    metadata=attention.install(model)
    joint=module('run042_composed_joint',RUN/'candidates'/parents['joint']/'joint.py')
    for layer in model.gpt_neox.layers:
        layer._run026_joint=joint.Joint(layer._run026_joint)
    head=module('run042_composed_head',RUN/'candidates'/parents['head']/'head.py')
    tile=read(RUN/'candidates'/parents['head']/'spec.json')['index']
    linear=model.embed_out
    linear._run042_head=head.Head(linear,tile)
    linear.forward=MethodType(lambda obj,x:obj._run042_head(x),linear)
    return {**metadata,'identity':'opt032','components':parents,
            'sparse_sites':['h','z'],'dense_head_change':True,
            'head_precision':'BF16 operands/output, FP32 accumulation, full50304 vocabulary'}
