"""Full native FP16 scalar T7 occupancy for the matched training comparison.

Read retained c36/c39 weights; no optimization, clipping or gate modification.
"""
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from run_config import RUN_DIR as RUN, load_config, require_validation_coverage, write_json
from optimizer_boundary import recipe_attention_context
from diagnostics import OccupancyAccumulator
from sparsity_research.capture import ActivationCapture
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.metrics import ActivationAccumulator
from sparsity_research.pythia import load_checkpoint_pythia

SITES=['a','m','h','q_post','k_post','v','z']

def identity(path):
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':path.relative_to(RUN).as_posix(),'bytes':path.stat().st_size,'sha256':digest}

def main():
    cfg=load_config(); inputs=json.loads((RUN/'latency/provenance/scalar-inputs.json').read_text())
    val=RUN/'latency'/inputs['validation']['path']
    actual=identity(val)
    assert actual['bytes']==inputs['validation']['bytes'] and actual['sha256']==inputs['validation']['sha256']
    tokens=np.memmap(val,dtype=np.int32,mode='r')
    device=torch.device('cuda'); torch.set_num_threads(2)
    started=time.monotonic()
    for row in inputs['checkpoints']:
        target=RUN/'artifacts/scalar-structure'/row['id']/'activation_statistics.json'
        if target.exists(): raise FileExistsError('Preserve completed post-hoc evidence')
        sources=[]
        for record in row['files']:
            path=RUN/'latency'/record['path']; source=identity(path)
            assert source['bytes']==record['bytes'] and source['sha256']==record['sha256']
            sources.append(source)
        model=load_checkpoint_pythia(AutoModelForCausalLM,RUN/'latency'/row['checkpoint'],torch=torch)
        model=model.to(device).eval(); model.set_attn_implementation('sdpa'); model.config.use_cache=False
        accumulator=ActivationAccumulator(tuple(sorted(set([0.,.001,.01,row['dose']]))))
        occupancy=OccupancyAccumulator(cfg['diagnostics']['tile_shapes'])
        with ActivationCapture(model,SITES,torch=torch) as capture, recipe_attention_context(torch,device):
            def after_batch(_output,_count):
                accumulator.update(capture.activations,torch=torch)
                for name,value in capture.activations.items(): occupancy.update(name,value,torch=torch)
                capture.clear()
            coverage=evaluate_complete_blocks(model=model,tokens=tokens,block_size=2048,
                batch_size=cfg['validation']['batch_size'],device=device,torch=torch,np=np,
                autocast_dtype=torch.float16,after_batch=after_batch)
        require_validation_coverage(coverage,cfg)
        rows=accumulator.rows(); pooled=accumulator.pooled_by_site(); tiles=occupancy.pooled()
        assert len(rows)==42 and all(r['nonfinite']==0 for r in rows)
        for moments,counts in zip(pooled,tiles,strict=True):
            assert moments['name']==counts['name']
            hist=counts['row_nonzero_histogram']
            assert sum(i*n for i,n in enumerate(hist))==moments['total']-moments['exact_zero_count']
            assert counts['tiles']['8x16']['total']*128==moments['total']
        write_json(target,{'status':'complete','checkpoint':row['id'],'kappa':row['dose'],
            'precision':'native FP16 autocast with FP32 parameters; matches training validation',
            'coverage':coverage,'rows':rows,'pooled_by_site':pooled,
            'occupancy_rows':occupancy.rows(),'occupancy_pooled_by_site':tiles,
            'source_files':sources+[identity(val),identity(Path(__file__))],
            'elapsed_seconds':time.monotonic()-started})
        print(json.dumps({'checkpoint':row['id'],'coverage':coverage,'elapsed_seconds':time.monotonic()-started}),flush=True)
        del model,capture,accumulator,occupancy
        gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__': main()
