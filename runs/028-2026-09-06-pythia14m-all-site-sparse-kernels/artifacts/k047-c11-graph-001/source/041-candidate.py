"""Compose qualified K042 input tiles with qualified K039 output tiles."""
from common import RUN,module

output=module('run028_k047_output',RUN/'candidates/k039/candidate.py')


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k047_base',RUN/'candidates/k042/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:layer._run026_joint=output.Joint(layer._run026_joint)
    metadata['output_projections']='k047 composition: unchanged K039 M32N32 output MMA tiles'
    return metadata
