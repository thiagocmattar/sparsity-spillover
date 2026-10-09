"""Retain the exact real pre-gate operands for timing and later profiling."""
import argparse
import time
import numpy as np
import torch
from common import RUN, REPO, config, sources, verify, record, write, runtime, load_model, manifest_identity, check_runtime, emit


def main():
    p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--inputs',type=int)
    args=p.parse_args();cfg=config();runtime(cfg['runtime_seed']);identity=manifest_identity(args.development)
    actual_runtime=check_runtime(args.development);inputs=sources()
    tokens=np.memmap(verify(inputs['validation']),dtype=np.int32,mode='r')
    if divmod(len(tokens),2048)!=(338,1444):raise ValueError('Validation coverage mismatch')
    indices=np.random.default_rng(cfg['timing_seed']).choice(338,args.inputs or cfg['timing_inputs'],replace=False)
    dest=RUN/('prelaunch/development-capture' if args.development else 'artifacts/capture')
    dest.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    captured=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for source in inputs['checkpoints']:
            model=load_model(REPO/source['checkpoint']);values=[];token_rows=[];current={}
            def hook(_module,arguments):current['value']=arguments[0].detach().clone()
            handle=model.head_input_gate.register_forward_pre_hook(hook)
            for j,index in enumerate(indices):
                ids=torch.tensor(tokens[index*2048:(index+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                model(input_ids=ids,use_cache=False)
                values.append(current.pop('value').cpu());token_rows.append(ids.cpu())
                emit(dest,'capture',model=source['id'],input=j+1,total=len(indices),elapsed_seconds=time.monotonic()-started)
            handle.remove()
            path=dest/(source['id']+'.pt')
            torch.save({'pre':values,'ids':token_rows,'weight':model.embed_out.weight.detach().cpu(),
                        'gate':model.config.head_input_gate,'source':source,'indices':indices.tolist()},path)
            captured.append(record(path));del model,values
            torch.cuda.empty_cache()
    write(dest/'manifest.json',{'status':'complete','identity':identity,'runtime':actual_runtime,
          'source':record(REPO/'runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/inference/provenance/trained-inputs.json'),
          'indices':indices.tolist(),'files':captured,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
