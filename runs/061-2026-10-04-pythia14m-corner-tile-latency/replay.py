"""Frozen K050 dense components with the approved run-local h/z consumer."""
import sys
from io_utils import RUN, ARCHIVE, archived_run, module

R25, R26, R27, R28 = [archived_run(n) for n in [25,26,27,28]]
sys.path[:0] = [str(R28),str(R27),str(R25),str(ARCHIVE/'src')]
common = module('common',R28/'common.py')
dense = common.dense
scaffold = module('run061_frozen_graph',R28/'45_graph_forward.py')


def install(model, mode):
    if mode not in {'sparse','no_skip'}:
        raise ValueError(mode)
    from kernel import gate_spec, Joint
    gates = [(gate_spec(layer.mlp.act),gate_spec(layer.attention.z_gate))
             for layer in model.gpt_neox.layers]
    base = module('run061_hz_base',RUN/'hz_adapter.py')
    base.install(model,'sparse')
    candidate = module('run061_k050',R28/'candidates/k050/candidate.py')
    metadata = candidate.install(model,shortcut=False,round_p=False,skip=True,projection_skip=True)
    from types import MethodType
    for layer,(gh,gz) in zip(model.gpt_neox.layers,gates):
        layer._run026_joint = Joint(layer._run026_joint,gh,gz,skip=mode=='sparse')
        layer.mlp.act.forward = MethodType(lambda _self,value:value,layer.mlp.act)
        layer.attention.z_gate.forward = MethodType(lambda _self,value:value,layer.attention.z_gate)
    metadata.update(output_projections='Run061 K049-derived M8/K16 h/z consumer; scalar or causal corner gate fused',
                    gate_specs=gates,hz_skip=mode=='sparse',
                    control='same gate and dense components; no_skip disables only h/z tile bypass and short-row substitution')
    return metadata
