"""Real-data memory, identity, end-to-end timing and complete-validation preflight."""
import argparse
import json
import math
import statistics
import sys
from time import perf_counter
from run_config import (RUN_DIR, load_config, load_verified_caches, build_schedule,
    condition_specs, resolved_condition_config, seed_everything, parameter_sha256,
    microbatches_for_step, EXPECTED_INITIAL_PARAMETER_SHA256, EXPECTED_SCHEDULE_SHA256,
    EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256, run_code_identity, write_json)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);args=p.parse_args()
    import numpy as np
    import torch
    import transformers
    from sparsity_research.pythia import topology_metadata
    from model_factory import build_pinned_run043_model
    from sparsity_research.pressure import parse_pressure_config
    from sparsity_research.optimization import set_learning_rate
    from initialization_artifact import load_pinned_initialization
    from optimizer_boundary import build_recipe_adamw,DynamicLossScaler,run_recipe_boundary,recipe_learning_rate
    from run043_capture import HOnlyPressureCapture
    from training import timed_validation
    from diagnostics import activation_diagnostic_validation,logical_product_validation
    c=load_config();t=c['training'];device=torch.device('cuda')
    out=RUN_DIR/'prelaunch'/('remote-preflight-'+args.worker+'.json')
    runtime={'python':f'{sys.version_info.major}.{sys.version_info.minor}',
             'torch':torch.__version__.split('+')[0],'transformers':transformers.__version__,
             'cuda_runtime':torch.version.cuda}
    if runtime!=c['runtime']:raise RuntimeError(f'Runtime mismatch: {runtime}')
    train,val,tm,vm,cache_seconds=load_verified_caches(c,np=np)
    starts,schedule_hash,schedule=build_schedule(c,tm,np=np)
    if schedule_hash!=EXPECTED_SCHEDULE_SHA256:raise RuntimeError('Data order mismatch')
    condition=next(x for x in condition_specs(c) if x['id']==args.worker)
    resolved=resolved_condition_config(c,condition)
    seed_everything(torch,c['seeds']['model'])
    model=build_pinned_run043_model(resolved['model'],device=torch.device('cpu'),torch=torch,auto_model=transformers.AutoModelForCausalLM)
    seed_everything(torch,c['seeds']['model'])
    initialization=load_pinned_initialization(model,torch=torch)
    initial_hash=parameter_sha256(model)
    result={'kind':'non_evidence_real_data_preflight','status':'running','worker':args.worker,
        'runtime':runtime,'gpu':torch.cuda.get_device_name(),
        'initial_parameter_sha256':initial_hash,'initialization':initialization,
        'schedule_sha256':schedule_hash,'topology':topology_metadata(model),
        'cache_verification_seconds':cache_seconds,'run_code':run_code_identity()}
    write_json(out,result)
    if initial_hash!=EXPECTED_INITIAL_PARAMETER_SHA256:raise RuntimeError('Initialization mismatch')
    model.to(device=device,dtype=torch.float32)
    optimizer,_=build_recipe_adamw(model,t,torch=torch)
    scaler=DynamicLossScaler(scale=2.**t['initial_loss_scale_power'],
        growth_interval=t['loss_scale_window'],hysteresis=t['loss_scale_hysteresis'],minimum_scale=t['minimum_loss_scale'])
    pressure=parse_pressure_config(resolved['activation_pressure'])
    torch.cuda.reset_peak_memory_stats();times=[];health=[]
    with HOnlyPressureCapture(model,['h'],torch=torch) as capture:
        for step in range(1,7):
            model.train();torch.cuda.synchronize();start=perf_counter()
            batches=microbatches_for_step(train,starts[step-1],block_size=2048,device=device,torch=torch,np=np)
            set_learning_rate(optimizer,recipe_learning_rate(step,peak=t['peak_learning_rate'],max_steps=t['max_steps'],warmup_fraction=t['warmup_fraction'],minimum=t['minimum_learning_rate']))
            row=run_recipe_boundary(model=model,optimizer=optimizer,batches=batches,pressure=pressure,
                capture=capture,loss_scaler=scaler,gradient_clip_norm=t['gradient_clip_norm'],torch=torch,device=device)
            torch.cuda.synchronize();elapsed=perf_counter()-start
            del batches
            if row['optimizer_step_skipped'] or row['gradient_overflow']:raise RuntimeError('Skipped preflight update')
            if row['pressure_capture_tensor_count']!=6 or row['pressure_capture_names_sha256']!=EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256:raise RuntimeError('Capture mismatch')
            if row['pressure_to_task_ratio_final']>1+1e-9 or not math.isfinite(row['task_loss']):raise RuntimeError('Invalid boundary')
            if step>1:times.append(elapsed)
            health.append(row)
            print(json.dumps({'phase':'preflight_boundary','worker':args.worker,'step':step,'loss':row['task_loss'],'seconds':elapsed}),flush=True)
    validation,validation_seconds=timed_validation(model=model,tokens=val,config=c,torch=torch,np=np)
    start=perf_counter();checkpoint=RUN_DIR/'prelaunch'/('calibration-checkpoint-'+args.worker)
    model.save_pretrained(checkpoint,safe_serialization=True);checkpoint_seconds=perf_counter()-start
    diagnostic_seconds={}
    if condition['order']==4:
        start=perf_counter();activation_diagnostic_validation(model=model,tokens=val,config=c,torch=torch,np=np)
        diagnostic_seconds['activation']=perf_counter()-start
        start=perf_counter();logical_product_validation(model=model,tokens=val,config=c,torch=torch,np=np)
        diagnostic_seconds['logical']=perf_counter()-start
    reserved=torch.cuda.max_memory_reserved();capacity=torch.cuda.get_device_properties(0).total_memory
    result.update(status='passed' if reserved<=.9*capacity else 'failed',
        end_to_end_boundary_seconds=times,median_boundary_seconds=statistics.median(times),
        boundary_health=health,validation=validation,validation_seconds=validation_seconds,
        checkpoint_seconds=checkpoint_seconds,diagnostic_seconds=diagnostic_seconds,
        peak_reserved_bytes=reserved,total_memory_bytes=capacity,
        predicted_training_seconds=statistics.median(times)*t['max_steps'])
    write_json(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'run_code','boundary_health'}}),flush=True)
    if result['status']!='passed':raise RuntimeError('Insufficient memory headroom')


if __name__=='__main__':main()