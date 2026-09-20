"""Untimed trace of the already qualified final T7/Ph selection; no selection feedback."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import transformers
import replay
from profiling import collect
from io_utils import RUN,read,record,verify,write
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    finished=read(RUN/'artifacts/optimized/summary-001.json')
    assert len(finished)==6 and all(x['qualified'] for x in finished)
    selection=read(RUN/'provenance/final-selection.json')
    selected=selection['candidate'];verify(selection['manifest'])
    cfg=read(RUN/'config.json');manifest=read(RUN/'provenance/inputs.json')
    checkpoint=next(x for x in manifest['checkpoints'] if x['id']=='c21')
    for item in checkpoint['files']+checkpoint['provenance']:verify(item)
    assert torch.cuda.get_device_name()==cfg['gpu']
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    data=np.memmap(verify(manifest['validation']),dtype=np.int32,mode='r')
    indices=read(RUN/'provenance/timing-indices.json')[:4]
    destination=RUN/'artifacts/optimized-profiles/c21'
    destination.mkdir(parents=True,exist_ok=False)
    models={};runners={}
    with torch.inference_mode():
        inputs=[torch.tensor(data[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        for mode in ('native','candidate'):
            model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa');model.config.use_cache=False
            if mode=='candidate':replay.install(model,selected)
            models[mode]=model
            runner=replay.dense.DenseRunner(lambda ids,m=model:replay.scaffold.forward(m,ids),inputs[0].clone(),'graph')
            runner.prepare();runners[mode+'_graph']=runner
        collect(models,runners,inputs,destination)
    write(destination/'result.json',{'status':'complete','candidate':selected,'condition':'c21',
          'selection':record(RUN/'provenance/final-selection.json'),'script':record(__file__),
          'inputs':indices,'use':'Post-selection structural evidence only; all six final evaluations passed before this trace. No retuning permitted.'})


if __name__=='__main__':main()
