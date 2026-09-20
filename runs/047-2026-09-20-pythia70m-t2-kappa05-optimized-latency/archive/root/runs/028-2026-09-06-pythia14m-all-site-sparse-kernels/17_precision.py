"""Development-only attribution of attention rounding and z-gate differences."""
import argparse
import json
import time
import traceback
import torch
import transformers
import numpy as np
from torch.nn.attention import SDPBackend,sdpa_kernel
from common import RUN,ROOT,R27,module,manifest,runtime,verify_record,record,write_json
from sparsity_research.pythia import load_checkpoint_pythia

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--condition',default='c30');p.add_argument('--inputs',type=int,default=2);a=p.parse_args()
    if a.condition not in {'c25','c30'} or not 1<=a.inputs<=8 or not a.attempt.replace('-','').isalnum():p.error('Invalid development scope')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    try:
        runtime(torch);source=manifest();checkpoint=next(r for r in source['checkpoints'] if r['id']==a.condition)
        for row in checkpoint['files']+checkpoint['provenance']:verify_record(row)
        cache=np.memmap(verify_record(source['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
        previous=module('run028_precision_previous',R27/'adapter.py')
        candidate=module('run028_precision_candidate',RUN/'candidates/k022/candidate.py')
        net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
        net.set_attn_implementation('sdpa');net.config.use_cache=False
        previous.install(net,'sparse')
        captured={};strides={};handles=[]
        for index,layer in enumerate(net.gpt_neox.layers):
            for site in ['q_post','k_post','v','z']:
                def hook(mod,args,out,index=index,site=site):captured[index,site]=out.detach().clone();strides[index,site]=list(out.stride())
                handles.append(getattr(layer.attention,site+'_site').register_forward_hook(hook))
        write_json(dest/'manifest.json',{'arguments':vars(a),'checkpoint':checkpoint,'inputs':source['inputs'],
            'sources':[record(RUN/name) for name in ['17_precision.py','common.py','candidates/k022/kernel.cu','candidates/k022/candidate.py']],
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'scope':'development component diagnosis; no timing claim'})
        with torch.inference_mode():
            for input_index in range(a.inputs):
                ids=torch.tensor(cache[input_index].copy(),device='cuda',dtype=torch.long)[None]
                net(input_ids=ids,use_cache=False)
                for index,layer in enumerate(net.gpt_neox.layers):
                    q,k,v=[captured[index,s] for s in ['q_post','k_post','v']]
                    ref=captured[index,'z'].reshape(1,2048,4,32).transpose(1,2).contiguous()
                    scale=layer.attention.scaling;gate=layer.attention.z_gate.kappa
                    modes={};errors={}
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                        modes['sdpa_default']=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
                    operators=[event.key for event in prof.key_averages() if 'attention' in event.key]
                    for backend in ['FLASH_ATTENTION','EFFICIENT_ATTENTION','MATH','CUDNN_ATTENTION']:
                        try:
                            with sdpa_kernel(getattr(SDPBackend,backend)):
                                modes[backend]=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
                        except RuntimeError as exc:errors[backend]=str(exc)
                    for round_p in [False,True]:
                        for prefix in [False,True]:modes[f'k022-prefix{int(prefix)}-round{int(round_p)}']=candidate.Attention(prefix,round_p)(q,k,v,scale).clone()
                    comparisons={}
                    for mode,out in modes.items():
                        delta=out.float()-ref.float();changed=(out>=gate)!=(ref>=gate)
                        pairs=torch.stack([ref[changed].float(),out[changed].float()],dim=-1)
                        comparisons[mode]={'bitwise_equal':torch.equal(out,ref),'max_abs':float(delta.abs().max()),
                            'relative_l2':float(delta.norm()/ref.float().norm().clamp_min(1e-30)),
                            'gate_mismatches':int(changed.sum()),'gate_mismatch_value_pairs':pairs[:64].cpu().tolist(),
                            'post_gate_max_abs':float((out.masked_fill(out<gate,0).float()-ref.masked_fill(ref<gate,0).float()).abs().max())}
                    row={'input':input_index,'layer':index,'strides':{s:strides[index,s] for s in ['q_post','k_post','v','z']},
                        'threshold':gate,'scale':scale,'operators':operators,'reference_gate_survivors':int((ref>=gate).sum()),'comparisons':comparisons,'backend_errors':errors}
                    rows.append(row);write_json(dest/'precision.json',rows)
                    print(json.dumps({'input':input_index,'layer':index,'operators':operators,'gate_mismatches':{k:v['gate_mismatches'] for k,v in comparisons.items()}}),flush=True)
        for handle in handles:handle.remove()
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
