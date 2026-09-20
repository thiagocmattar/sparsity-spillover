"""Untimed localization of native-versus-graph numerical divergence."""
import argparse
import torch
import numpy as np
import transformers
from common import ROOT,RUN,HERE,dense,inputs_manifest,verify_record,write_json,numerical_gate
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    manifest,row=inputs_manifest()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/row['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
    model.set_attn_implementation('sdpa')
    model.config.use_cache=False
    data=np.memmap(verify_record(manifest['development']),dtype=np.int32,mode='r').reshape(64,2048)
    inputs=[torch.tensor(a.copy(),device='cuda',dtype=torch.long)[None] for a in data[:16]]
    state={}
    def hook(name):
        def capture(mod,inputs,output):
            state[name]=output.clone()
        return capture
    for i,layer in enumerate(model.gpt_neox.layers):
        modules={'a':layer.input_layernorm,'m':layer.post_attention_layernorm,
            'W1':layer.mlp.dense_h_to_4h,'h':layer.mlp.act,'W2':layer.mlp.dense_4h_to_h,
            'QKV':layer.attention.query_key_value,'Wo':layer.attention.dense}
        modules.update({s:getattr(layer.attention,f'{s}_site') for s in ['q_pre','k_pre','q_post','k_post','v','z']})
        for name,mod in modules.items():mod.register_forward_hook(hook(f'L{i}.{name}'))
    with torch.inference_mode():
        runner=dense.DenseRunner(lambda ids:model(ids,use_cache=False).logits,inputs[0].clone(),'graph')
        runner.prepare()
        graph_state=dict(state)
        state.clear()
        rows=[]
        for i,ids in enumerate(inputs):
            state.clear()
            expected=model(ids,use_cache=False).logits
            native_state=dict(state)
            runner.stage(ids)
            actual=runner()
            torch.cuda.synchronize()
            differences=[]
            for name,expected_activation in native_state.items():
                other=graph_state[name]
                if not torch.equal(expected_activation,other):
                    differences.append({'site':name,'max_abs':float((expected_activation.float()-other.float()).abs().max()),
                        'different_elements':int((expected_activation!=other).sum()),
                        'zero_mask_changes':int(((expected_activation==0)!=(other==0)).sum())})
            rows.append({'input':i,'differences':differences,
                'logits':numerical_gate(expected,actual,relative_l2=.02,atol=.25,rtol=.02)})
    write_json(RUN/'runtime'/f'{args.name}.json',{'untimed_diagnostic':True,'rows':rows})
    print([{'input':r['input'],'first_difference':r['differences'][:1],'logit_pass':r['logits']['pass']} for r in rows],flush=True)


if __name__=='__main__':main()
