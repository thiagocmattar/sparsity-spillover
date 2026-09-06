"""Corrected attention-only composition: K036 projections plus K044 attention."""
from common import RUN,module

parent=module('run028_k046_attention',RUN/'candidates/k044/candidate.py')
Attention=parent.Attention
extension=parent.extension


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k046_base',RUN/'candidates/k036/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:layer.attention._run028_attention=Attention(skip,shortcut)
    metadata['attention']='k046 corrected K036 projection composition; K044 warp-uniform predicated BF16 MMA'
    return metadata
