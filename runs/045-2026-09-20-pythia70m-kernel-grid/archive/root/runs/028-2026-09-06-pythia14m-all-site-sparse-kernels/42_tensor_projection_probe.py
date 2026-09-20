"""Attribute K032 a/m rounding on native operands from known failing blocks.

This is a selected-case diagnostic, not independent qualification or timing.
FP64 dots distinguish bias-rounding conventions from accumulation-order ties.
"""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN, ROOT, R27, manifest, module, runtime, record, verify_record, write_json
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity required')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    def emit(stage,**fields):
        row={'stage':stage,'elapsed_seconds':time.monotonic()-started,**fields}
        write_json(dest/'status.json',row)
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        runtime(torch);source=manifest()
        paths=[RUN/'42_tensor_projection_probe.py',RUN/'common.py']+list((RUN/'candidates/k032').glob('*.py'))+list((RUN/'candidates/k032').glob('*.cu'))
        for path in paths:
            target=dest/'source'/path.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(dest/'manifest.json',{'arguments':vars(a),'sources':[record(f) for f in paths],
            'validation':source['inputs']['validation'],'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'scope':'c25 validation blocks74/118/119, c30 block0; selected failure attribution only'})
        projection=module('run028_tensor_projection_precision',RUN/'candidates/k032/candidate.py')
        original=module('run028_tensor_projection_precision_original',R27/'adapter.py')
        data=np.memmap(verify_record(source['inputs']['validation']),dtype=np.int32,mode='r')
        with torch.inference_mode():
            projection.extension()
            for n in [384,512]:
                linear=torch.nn.Linear(128,n,device='cuda',dtype=torch.bfloat16).eval()
                for zero in [0.,.75,1.]:
                    x=torch.randn((1,2048,128),device='cuda',dtype=torch.bfloat16)
                    x.masked_fill_(torch.rand_like(x.float())<zero,0)
                    ref=linear(x);out=projection.Projection(linear)(x).clone()
                    no_skip=projection.Projection(linear,skip=False)(x).clone()
                    torch.testing.assert_close(out,ref,atol=.125,rtol=.02)
                    if not torch.equal(out,no_skip):raise ValueError('Synthetic skip toggle mismatch')
                    rows.append({'synthetic':True,'n':n,'zero_fraction_target':zero,
                                 'different_elements':int((out!=ref).sum()),'max_abs':float((out.float()-ref.float()).abs().max())})
            del linear,x,out,no_skip
            for condition,blocks in [('c25',[74,118,119]),('c30',[0])]:
                checkpoint=next(r for r in source['checkpoints'] if r['id']==condition)
                for row in checkpoint['files']+checkpoint['provenance']:verify_record(row)
                write_json(dest/f'checkpoint-{condition}.json',checkpoint)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False;original.install(net,'fusion_dense')
                captured={};handles=[];linears={}
                for index,layer in enumerate(net.gpt_neox.layers):
                    for site,linear in [('a',layer.attention.query_key_value),('m',layer.mlp.dense_h_to_4h)]:
                        linears[index,site]=linear
                        def hook(mod,args,out,index=index,site=site):captured[index,site]=(args[0].detach().clone(),out.detach().clone())
                        handles.append(linear.register_forward_hook(hook))
                if condition=='c25':
                    dispatch=[]
                    for index,layer in enumerate(net.gpt_neox.layers[:1]):
                        for site,linear in [('a',layer.attention.query_key_value),('m',layer.mlp.dense_h_to_4h),
                                            ('h',layer.mlp.dense_4h_to_h),('z',layer.attention.dense)]:
                            x=torch.randn((1,2048,linear.weight.shape[1]),device='cuda',dtype=torch.bfloat16)
                            torch.nn.functional.linear(x,linear.weight,linear.bias);torch.cuda.synchronize()
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                                torch.nn.functional.linear(x,linear.weight,linear.bias);torch.cuda.synchronize()
                            prof.export_chrome_trace(str(dest/f'native-{site}.json'))
                            dispatch.append({'site':site,'cuda_events':[{'name':e.name,'device_time_total':e.device_time_total}
                                             for e in prof.events() if 'CUDA' in str(e.device_type)]})
                    write_json(dest/'dispatch.json',dispatch)
                for block in blocks:
                    ids=torch.tensor(data[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                    net(input_ids=ids,use_cache=False)
                    for (index,site),(x,ref) in captured.items():
                        linear=linears[index,site];fn=projection.Projection(linear)
                        out=fn(x).clone();no_skip=projection.Projection(linear,skip=False)(x).clone()
                        if not torch.equal(out,no_skip):raise ValueError('Sparse/dense tensor projection mismatch')
                        xf=x.reshape(-1,128);rf=ref.reshape(-1,ref.shape[-1]);of=out.reshape_as(rf)
                        mismatch=(of!=rf).nonzero();take=mismatch[:256]
                        # A second reference distinguishes a separately rounded BF16 matmul/bias.
                        two_round=torch.nn.functional.linear(x,linear.weight,None)+linear.bias
                        details=[]
                        if take.numel():
                            xr=xf[take[:,0]].double();wr=linear.weight[take[:,1]].double();br=linear.bias[take[:,1]].double()
                            exact=(xr*wr).sum(1)+br;rounded=exact.bfloat16()
                            for j,(r,c) in enumerate(take.cpu().tolist()):
                                details.append({'row':r,'channel':c,'native':float(rf[r,c]),'sparse':float(of[r,c]),
                                    'fp64_dot_plus_bias':float(exact[j]),'fp64_rounded':float(rounded[j]),
                                    'separate_bf16_bias':float(two_round.reshape_as(rf)[r,c])})
                        row={'condition':condition,'validation_block':block,'layer':index,'site':site,
                            'elements':out.numel(),'different_elements':len(mismatch),'max_abs':float((out.float()-ref.float()).abs().max()),
                            'native_equals_separate_bf16_bias':torch.equal(two_round,ref),
                            'one_sided_gate_mismatches_0p5':int(((out>=.5)!=(ref>=.5)).sum()),
                            'selected_differences_limit':256,'differences':details}
                        rows.append(row);write_json(dest/'components.json',rows)
                    emit('block',condition=condition,validation_block=block,cases=len(rows))
                for h in handles:h.remove()
                del net,captured,linears;torch.cuda.empty_cache()
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'elapsed_seconds':time.monotonic()-started})
        emit('complete',cases=len(rows))
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});emit('failed',error=str(exc));raise


if __name__=='__main__':main()
