"""Untimed component substitutions to locate the failed A0 logit qualification.

No timing result or kernel selection by speed is made by this diagnostic.
"""
import gc
import json
from types import MethodType
import numpy as np
import torch
import transformers
import replay
from io_utils import RUN,read,write
from sparsity_research.pythia import load_checkpoint_pythia

class NativeJoint:
    def __init__(self,old):self.old=old
    def __call__(self,h,z,residual):
        o=self.old
        if o.gh:h=h.masked_fill(h<o.th,0)
        if o.gz:z=z.masked_fill(z<o.tz,0)
        return (torch.nn.functional.linear(h,o.w2.weight,o.w2.bias)+
                torch.nn.functional.linear(z,o.wo.weight,o.wo.bias))+residual

def native_attention(q,k,v,scale):
    return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)

def substitute(model,components):
    for layer in model.gpt_neox.layers:
        if 'norm' in components:
            for norm in [layer.input_layernorm,layer.post_attention_layernorm]:
                norm.forward=MethodType(torch.nn.LayerNorm.forward,norm)
        if 'projections' in components:
            for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
                linear.forward=MethodType(torch.nn.Linear.forward,linear)
        if 'joint' in components:layer._run026_joint=NativeJoint(layer._run026_joint)
        if 'attention' in components:layer.attention._run028_attention=native_attention

def main():
    cfg=read(RUN/'config.json');row=read(RUN/'provenance/inputs.json')['checkpoints'][0]
    policy=read(RUN/'provenance/candidates.json')['configurations'][0]
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    val=np.memmap(RUN/'inputs/validation.int32.bin',dtype=np.int32,mode='r')
    blocks=[16,21,99]
    ids=[torch.tensor(val[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in blocks]
    def load():
        model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/row['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
        model.set_attn_implementation('sdpa');model.config.use_cache=False
        return model
    all_components={'norm','projections','joint','attention'}
    cases=[('full_port',set())]+[(f'native_{x}',{x}) for x in sorted(all_components)]
    cases += [(f'only_port_{x}',all_components-{x}) for x in sorted(all_components)]
    cases += [('all_native_components',all_components)]
    results=[]
    with torch.inference_mode():
        model=load();refs=[model(input_ids=x,use_cache=False).logits.clone() for x in ids];del model
        for name,components in cases:
            model=load();replay.install(model,policy);substitute(model,components)
            rows=[]
            for index,x,ref in zip(blocks,ids,refs):
                actual=replay.scaffold.forward(model,x)
                delta=actual.float()-ref.float()
                tol=.25+.02*ref.float().abs()
                rows.append({'block':index,'max_abs':float(delta.abs().max()),
                    'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float())),
                    'violating_logits':int((delta.abs()>tol).sum()),'exact':bool(torch.equal(actual,ref))})
                del actual,delta,tol
            result={'case':name,'native_components':sorted(components),'blocks':rows}
            print(json.dumps(result),flush=True);results.append(result)
            write(RUN/'artifacts/numerical-isolation-001.json',results)
            del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
