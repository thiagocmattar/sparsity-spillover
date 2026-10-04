"""Frozen K050 dense components with the approved run-local h/z consumer."""
import sys
from types import MethodType
from io_utils import RUN, ARCHIVE, archived_run, module

R25, R26, R27, R28 = [archived_run(n) for n in [25,26,27,28]]
# Run-local modules must win over historical modules with the same basename.
sys.path.extend([str(R28),str(R27),str(R25),str(ARCHIVE/'src')])
common = module('common',R28/'common.py')
dense = common.dense
scaffold = module('run062_frozen_graph',R28/'45_graph_forward.py')


def install(model, mode):
    if mode not in {'sparse','no_skip'}:
        raise ValueError(mode)
    from kernel import gate_spec, Joint
    gates = [(gate_spec(layer.mlp.act),gate_spec(layer.attention.z_gate))
             for layer in model.gpt_neox.layers]
    base = module('run062_hz_base',RUN/'hz_adapter.py')
    base.install(model,'sparse')
    candidate = module('run062_k049',R28/'candidates/k049/candidate.py')
    metadata = candidate.install(model,shortcut=False,round_p=False,skip=True,projection_skip=True)
    from port_gates import NormPair
    for layer in model.gpt_neox.layers:
        pair = NormPair(layer)
        layer.input_layernorm.forward = MethodType(lambda _self,x,p=pair:p.start(x),layer.input_layernorm)
        layer.post_attention_layernorm.forward = MethodType(lambda _self,x,p=pair:p.finish(x),layer.post_attention_layernorm)
        layer.a_gate.forward = MethodType(lambda _self,x:x,layer.a_gate)
        layer.m_gate.forward = MethodType(lambda _self,x:x,layer.m_gate)
        layer._run028_norm_pair = pair
    for layer,(gh,gz) in zip(model.gpt_neox.layers,gates):
        layer._run026_joint = Joint(layer._run026_joint,gh,gz,skip=mode=='sparse')
        layer.mlp.act.forward = MethodType(lambda _self,value:value,layer.mlp.act)
        layer.attention.z_gate.forward = MethodType(lambda _self,value:value,layer.attention.z_gate)
    metadata.update(output_projections='Run062 K049-derived M8/K16 h/z consumer; scalar or causal corner gate fused',
                    normalization='unchanged K050 rounded norm pair, followed by one exact a/m gate pass',
                    qkv_postprocessing='unchanged unthresholded K019 RoPE producer, followed by one exact headwise q/k/v gate pass',
                    common_gate_binary='port_gates.cu, scalar/corner mode selected at runtime',
                    gate_specs=gates,hz_skip=mode=='sparse',
                    control='same gate and dense components; no_skip disables only h/z tile bypass and short-row substitution')
    return metadata
