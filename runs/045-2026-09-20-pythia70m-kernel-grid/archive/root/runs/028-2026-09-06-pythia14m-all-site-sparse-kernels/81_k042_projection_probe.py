"""Six synthetic plus48 captured training cases for wider a/m projections."""
import argparse
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN, ROOT, R27, manifest, module, runtime, record, verify_record, write_json, read_json
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args()
    if not args.attempt.replace('-','').isalnum():parser.error('Simple attempt identity required')
    dest=RUN/'artifacts'/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    try:
        runtime(torch);source=manifest();frozen=read_json(RUN/'final-policy-001.json')
        for entry in frozen['sources']+frozen['dependencies']['files']:verify_record(entry)
        paths=sorted(set([ROOT/entry['path'] for entry in frozen['sources']]+[RUN/'81_k042_projection_probe.py']+
            [p for p in (RUN/'candidates/k042').iterdir() if p.is_file()]))
        snapshots=[]
        for index,path in enumerate(paths):
            target=dest/'source'/f'{index:03d}-{path.name}';target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);snapshots.append({'snapshot':str(target.relative_to(dest)),'original':record(path)})
        write_json(dest/'manifest.json',{'arguments':vars(args),'sources':snapshots,'inputs':source['inputs'],
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':2801,
            'scope':'6 synthetic plus48 a/m cases from c25/c30 training blocks0/1; no full-model timing claim'})
        projection=module('run028_k042_probe',RUN/'candidates/k042/candidate.py')
        original=module('run028_k042_original',R27/'adapter.py')
        data=np.memmap(verify_record(source['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
        def case(identity,x,linear,reference):
            out=projection.Projection(linear,skip=True)(x).clone()
            unskipped=projection.Projection(linear,skip=False)(x).clone()
            row={'identity':identity,'shape':list(out.shape),'bitwise_native':torch.equal(out,reference),
                'bitwise_skip_toggle':torch.equal(out,unskipped),'different_elements':int((out!=reference).sum()),
                'max_abs':float((out.float()-reference.float()).abs().max())}
            rows.append(row);write_json(dest/'components.json',rows);print(row,flush=True)
            if not row['bitwise_native'] or not row['bitwise_skip_toggle']:raise ValueError('Projection arithmetic changed')
        with torch.inference_mode():
            projection.extension()
            for n in [384,512]:
                linear=torch.nn.Linear(128,n,device='cuda',dtype=torch.bfloat16).eval()
                for fraction in [0.,.75,1.]:
                    x=torch.randn((1,2048,128),device='cuda',dtype=torch.bfloat16)
                    x.masked_fill_(torch.rand_like(x.float())<fraction,0)
                    case(f'synthetic-n{n}-zero{fraction}',x,linear,linear(x))
            del linear,x
            for condition in ['c25','c30']:
                checkpoint=next(row for row in source['checkpoints'] if row['id']==condition)
                for entry in checkpoint['files']+checkpoint['provenance']:verify_record(entry)
                write_json(dest/f'checkpoint-{condition}.json',checkpoint)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False;original.install(net,'fusion_dense')
                captured={};linears={};handles=[]
                for layer_index,layer in enumerate(net.gpt_neox.layers):
                    for site,linear in [('a',layer.attention.query_key_value),('m',layer.mlp.dense_h_to_4h)]:
                        key=(layer_index,site);linears[key]=linear
                        def capture(mod,inputs,out,key=key):captured[key]=(inputs[0].detach().clone(),out.detach().clone())
                        handles.append(linear.register_forward_hook(capture))
                for block in [0,1]:
                    ids=torch.tensor(data[block].copy(),device='cuda',dtype=torch.long)[None]
                    net(input_ids=ids,use_cache=False)
                    for key,(x,reference) in captured.items():case(f'{condition}-train{block}-layer{key[0]}-{key[1]}',x,linears[key],reference)
                for handle in handles:handle.remove()
                del net,captured,linears;torch.cuda.empty_cache()
        if len(rows)!=54:raise ValueError('Incomplete component coverage')
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'bitwise_native_all':all(r['bitwise_native'] for r in rows),
            'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
