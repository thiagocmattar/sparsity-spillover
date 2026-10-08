"""Reduced local CUDA mechanics, never production training or fit evidence."""
import argparse
import gc
from time import perf_counter
import numpy as np
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
from run_config import (RUN_DIR,load_config,condition_specs,resolved_condition_config,
                        EXPECTED_INITIAL_PARAMETER_SHA256,parameter_sha256,write_json)
from initialization import apply_pythia_14m_initialization,_BASE
from sparsity_research.pythia import apply_activation_topology
from head_site import install_head,ActivationCapture,parse_pressure_config
from optimizer_boundary import run_recipe_boundary,DynamicLossScaler,build_recipe_adamw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag.isalnum():raise ValueError('Simple unique tag required')
    output=RUN_DIR/f'prelaunch/cuda-smoke-{args.tag}.json'
    if output.exists():raise FileExistsError(output)
    cfg=load_config();rows=[];started=perf_counter();device=torch.device('cuda')
    cache=np.memmap(RUN_DIR.parents[1]/'data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin',dtype=np.int32,mode='r')
    for condition in condition_specs(cfg):
        resolved=resolved_condition_config(cfg,condition)
        architecture=GPTNeoXConfig(**_BASE.EXPECTED_ARCHITECTURE)
        architecture._attn_implementation='sdpa'
        architecture.topology_id='HZ';architecture.site_gate=None
        architecture.site_gates=resolved['model']['site_gates']
        model=install_head(apply_activation_topology(GPTNeoXForCausalLM(architecture),torch=torch),resolved['model']['head_input_gate']).to(device)
        apply_pythia_14m_initialization(model,torch=torch)
        initial=parameter_sha256(model)
        if initial!=EXPECTED_INITIAL_PARAMETER_SHA256:raise ValueError('Random replay drift')
        optimizer,_=build_recipe_adamw(model,cfg['training'],torch=torch)
        pressure=parse_pressure_config(resolved['activation_pressure'])
        scaler=DynamicLossScaler(scale=2.**cfg['training']['initial_loss_scale_power'])
        torch.cuda.reset_peak_memory_stats();boundaries=[]
        with ActivationCapture(model,condition['pressure_sites'],torch=torch) as capture:
            for step in range(2):
                batches=[torch.tensor(cache[(2*step+i)*2048:(2*step+i+1)*2048].copy(),device=device,dtype=torch.long)[None] for i in range(2)]
                boundary=run_recipe_boundary(model=model,optimizer=optimizer,batches=batches,
                    pressure=pressure,capture=capture,loss_scaler=scaler,gradient_clip_norm=1.,torch=torch,device=device)
                if boundary['optimizer_step_skipped'] or not boundary['ol1_correction_applied'] or boundary['pressure_capture_tensor_count']!=7:
                    raise RuntimeError('Reduced FP16 OL1 mechanics failed')
                boundaries.append(boundary);del batches
        torch.cuda.synchronize()
        rows.append({'id':condition['id'],'initial_parameter_sha256':initial,'boundaries':boundaries,
                     'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        del model,optimizer,capture;gc.collect();torch.cuda.empty_cache()
    write_json(output,{'status':'passed','kind':'non_evidence_reduced_cuda_mechanics',
        'gpu':torch.cuda.get_device_name(),'precision':'dynamic FP16; FP32 parameters/moments',
        'micro_batch_size':1,'gradient_accumulation_steps':2,'sequence_length':2048,
        'rows':rows,'seconds':perf_counter()-started,
        'limitation':'Does not qualify production MB32/GAS32 memory fit, ETC or scientific endpoints'})
    print({'status':'passed','conditions':len(rows),'boundaries':4},flush=True)


if __name__=='__main__':main()
