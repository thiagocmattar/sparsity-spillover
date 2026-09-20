"""Compose shape-ported K050 components without mutating original archives."""
from pathlib import Path
from types import MethodType
from common import module
from sparsity_research.pythia import topology_metadata
import replay

HERE=Path(__file__).resolve().parent
norm=module('run035_norm',HERE/'norm/candidate.py')
joint=module('run035_joint',HERE/'joint/candidate.py')
projection=module('run035_projection',HERE/'projection/candidate.py')
attention=module('run035_attention',HERE/'attention/candidate.py')

def install(model,shortcut=False,round_p=False,skip=True,projection_skip=True):
    c=model.config
    if (c.hidden_size,c.intermediate_size,c.num_attention_heads,c.num_hidden_layers)!=(512,2048,8,6):
        raise ValueError('Retained Pythia70M architecture required')
    if shortcut or round_p or not c.use_parallel_residual:raise ValueError('Frozen policy mismatch')
    pairs=[norm.NormPair(layer.input_layernorm,layer.post_attention_layernorm,
        getattr(layer,'a_gate',None),getattr(layer,'m_gate',None)) for layer in model.gpt_neox.layers]
    original=module('run035_frozen_adapter',replay.R27/'adapter.py')
    original.install(model,'sparse')
    attn_forward=module('run035_frozen_attention_forward',replay.R28/'candidates/k020/candidate.py').attention_forward
    for layer,pair in zip(model.gpt_neox.layers,pairs):
        layer._run026_joint.skip=projection_skip
        layer._run026_joint=joint.Joint(layer._run026_joint)
        layer.attention._run028_attention=attention.Attention(skip=skip,shortcut=False)
        layer.attention.forward=MethodType(attn_forward,layer.attention)
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=projection.Projection(linear,skip=projection_skip)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
        norm.bind_pair(layer,pair)
    return {'identity':'k050-70m-v2','topology':topology_metadata(model),
        'normalization':'width512 shared-input Welford pair; distinct affine and existing a/m gates',
        'input_projections':'same CUTLASS32x64x64 pipeline; input width512',
        'output_projections':'M8N128 tiles across N512; H2048/Z512;128-bit h support masks; same <=2-row SIMT fallback',
        'attention':'K050 exact-zero MMA bypass in native H8 D64 M128N128 unsplit Flash schedule; no prefix shortcut',
        'rope':'unchanged shape-parametric K019 BF16 RoPE and symmetric gates',
        'projection_skip':projection_skip,'attention_skip':skip,'round_p':False,'shortcut':False,
        'new_optimization_search':False}
