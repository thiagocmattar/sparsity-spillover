"""Untimed stage comparison to isolate native eager/graph numerical differences."""
import argparse
import torch
import numpy as np
from io_utils import RUN, REPO, read, verify, load_model, configure_cuda, write
from head import install
from measurement import Graph, compare
from forward import forward


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    cfg=read(RUN/'config.json');inputs=read(RUN/'provenance/inputs.json');configure_cuda(cfg['runtime_seed'])
    cache=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
    ids=torch.tensor(cache[:2048].copy(),device='cuda',dtype=torch.long)[None]
    checkpoint=REPO/inputs['checkpoints'][0]['checkpoint']
    records={};outputs={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,cache_enabled=True):
        for mode in ('native_eager','native_graph','dense_eager','dense_graph'):
            model=load_model(checkpoint)
            if mode.startswith('dense'):install(model,'dense',cfg)
            stages={};handles=[]
            wanted={n:m for n,m in model.named_modules() if n.endswith(('embed_in','input_layernorm','query_key_value','q_post_site','z_site','z_gate','dense_h_to_4h','act','dense_4h_to_h','final_layer_norm','head_input_gate'))}
            for name,module in wanted.items():
                def hook(_m,_inputs,out,n=name):stages[n]=out.detach().clone()
                handles.append(module.register_forward_hook(hook))
            fn=lambda value,m=model:m(input_ids=value,use_cache=False).logits
            if mode.endswith('graph'):
                graph=Graph(lambda value,m=model:forward(m,value),[ids]);graph();torch.cuda.synchronize();output=graph.output.clone()
            else:output=fn(ids).clone()
            torch.cuda.synchronize();records[mode]=stages;outputs[mode]=output
            for handle in handles:handle.remove()
    result={'outputs':{mode:compare(outputs['native_eager'],out,cfg['numerical_bounds']) for mode,out in outputs.items()},'stages':{}}
    for mode,stages in records.items():
        result['stages'][mode]=[]
        for name,value in stages.items():
            ref=records['native_eager'][name]
            result['stages'][mode].append({'stage':name,'dtype':str(value.dtype),'exact':torch.equal(ref,value),
                'max_abs':float((value.float()-ref.float()).abs().max()),
                'zero_mask_mismatch':int(value.eq(0).ne(ref.eq(0)).sum())})
    write(RUN/f'prelaunch/trace-{args.tag}.json',result)
    print(result['outputs'],flush=True)


if __name__=='__main__':main()
