"""K024 tile-sparse attention + element-sparse K021/K018 projections."""
from types import MethodType
import torch
import triton
from common import RUN,module
ops=module('run028_k024_kernels',RUN/'candidates/k024/kernels.py')
base=module('run028_k024_base',RUN/'candidates/k020/candidate.py')

def extension():return None  # Triton compilation occurs on the first call.

class Attention:
    def __init__(self,shortcut=True,round_p=False):
        # Both flags are retained for the component harness. Online BF16-P is
        # mandatory in this candidate, so round_p is intentionally not a mode.
        self.shortcut=shortcut;self.shape=None
    def __call__(self,q,k,v,scale,*,count=False):
        if torch.is_grad_enabled() or q.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        b,h,t,d=q.shape
        if (b,d)!=(1,32) or not 0<t<=2048:raise ValueError('B1 D32 T<=2048')
        nc=triton.cdiv(t,256)
        if self.shape!=q.shape:
            self.prefix=torch.empty(q.shape,device=q.device,dtype=torch.float32)
            self.totals=torch.empty((h,nc,32),device=q.device,dtype=torch.float32)
            self.out=torch.empty_like(q);self.stats=torch.empty((h,t,3),device=q.device,dtype=torch.int64)
            self.shape=q.shape
        if self.shortcut:ops.prefix_chunks[(nc,h)](v,self.prefix,self.totals,t,nc,256,num_warps=4)
        ops.attention[(triton.cdiv(t,16),h)](q,k,v,self.prefix,self.totals,self.out,self.stats,t,scale,
            nc,triton.next_power_of_2(nc),256,16,128,self.shortcut,count,num_warps=4,num_stages=1)
        return self.out

def install(model,shortcut=True,round_p=False):
    projections=module('run028_k024_projections',RUN/'candidates/k021/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=Attention(shortcut,round_p)
        layer.attention.forward=MethodType(base.attention_forward,layer.attention)
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=projections.Projection(linear)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    return {'sparse_operations':['qkv_projection','mlp_w1','mlp_w2','attention_output_projection','qk_scores','probability_value'],
        'dense_operations':['lm_head'],'attention':'k024','output_projections':'k018','input_projections':'k021',
        'attention_skip_granularity':'zero query/K/V tiles, not arbitrary individual zeros',
        'attention_counter_unit':'issued tensor-core FMA-equivalent terms including causal padding',
        'probability_rounding':'BF16 unnormalized online; round_p flag is not an independent mode'}
