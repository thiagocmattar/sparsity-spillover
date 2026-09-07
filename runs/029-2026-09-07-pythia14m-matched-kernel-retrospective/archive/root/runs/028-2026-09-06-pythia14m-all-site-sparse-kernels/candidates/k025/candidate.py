"""Attribution control: identical direct attention wrapper, native dense SDPA.

This is not an all-six-site winner. It isolates wrapper/input-projection gains
from genuine sparse-attention gains in K022 onward.
"""
from types import MethodType
import torch
from common import RUN,module
base=module('run028_k025_base',RUN/'candidates/k020/candidate.py')

class Attention:
    def __call__(self,q,k,v,scale):
        return torch.nn.functional.scaled_dot_product_attention(q.contiguous(),k.contiguous(),v.contiguous(),is_causal=True,scale=scale)

def install(model,shortcut=True,round_p=False,skip=True):
    projections=module('run028_k025_projections',RUN/'candidates/k021/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=Attention()
        layer.attention.forward=MethodType(base.attention_forward,layer.attention)
        layer._run026_joint.skip=skip
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=projections.Projection(linear,skip=skip)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    return {'sparse_operations':['qkv_projection','mlp_w1','mlp_w2','attention_output_projection'] if skip else [],
        'dense_operations':['lm_head','qk_scores','probability_value'],
        'attention':'native SDPA via the same direct wrapper',
        'projection_skipping_enabled':skip,'purpose':'attribution control; not an all-site sparse candidate'}
