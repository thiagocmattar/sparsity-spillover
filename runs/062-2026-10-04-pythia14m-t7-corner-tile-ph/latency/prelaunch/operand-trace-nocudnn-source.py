"""Development-only eager operand trace to locate qualification mismatches."""
import sys
from io_utils import RUN, ARCHIVE, read, verify, write
sys.path.append(str(ARCHIVE/'src'))
import kernel
import diagnostics
import replay
from topology import register
register()
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from sparsity_research.pythia import load_checkpoint_pythia

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
inputs=read(RUN/'provenance/development-inputs.json')
row=next(c for c in inputs['checkpoints'] if c['id']=='c39')
val=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
ids=torch.tensor(val[:2048].copy(),device='cuda',dtype=torch.long)[None]
traces={}
with torch.inference_mode():
    for mode in ['native','sparse']:
        model=load_checkpoint_pythia(AutoModelForCausalLM,RUN/row['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
        model.set_attn_implementation('sdpa'); model.config.use_cache=False
        if mode=='sparse': replay.install(model,mode)
        trace={}; handles=[]
        def capture(name):
            def hook(_m,_i,out): trace[name]=(out[0] if isinstance(out,tuple) else out).clone()
            return hook
        for i,layer in enumerate(model.gpt_neox.layers):
            modules={'a':layer.a_gate,'m':layer.m_gate,'q':layer.attention.q_post_site,
                'k':layer.attention.k_post_site,'v':layer.attention.v_site,'qkv':layer.attention.query_key_value,
                'up':layer.mlp.dense_h_to_4h,'z_raw':layer.attention.z_gate,'layer':layer}
            for name,module in modules.items(): handles.append(module.register_forward_hook(capture(f'{i}/{name}')))
            if mode=='sparse':
                original=layer._run026_joint
                def joint_trace(h,z,residual,i=i,original=original):
                    trace[f'{i}/ref_joint']=torch.nn.functional.linear(kernel.gated_value(h,original.gh,original.th),original.w2.weight,original.w2.bias)+torch.nn.functional.linear(kernel.gated_value(z,original.gz,original.tz),original.wo.weight,original.wo.bias)+residual
                    trace[f'{i}/z_raw']=kernel.gated_value(z,original.gz,original.tz).clone()
                    return original(h,z,residual)
                layer._run026_joint=joint_trace
        trace['logits']=replay.scaffold.forward(model,ids).clone()
        traces[mode]=trace
        for handle in handles: handle.remove()
    rows=[]
    for name,a in traces['native'].items():
        b=traces['sparse'][name]; diff=(a.float()-b.float()).abs()
        rows.append({'name':name,'max_abs':diff.max().item(),'different':(a!=b).sum().item(),
                     'zero_mask_differences':((a==0)!=(b==0)).sum().item(),
                     'relative_l2':(diff.norm()/a.float().norm()).item()})
    write(RUN/'prelaunch/operand-trace.json',rows)
    for r in rows: print(r,flush=True)
    for i in range(6):
        # Canonical parallel residual is attention+MLP, then residual.
        a=traces['sparse'][f'{i}/ref_joint']; b=traces['sparse'][f'{i}/layer']
        print('joint reference',i,(a-b).abs().max().item(),(a!=b).sum().item(),flush=True)
