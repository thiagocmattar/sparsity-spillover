"""Non-evidence CPU mechanics: real 14M architecture, T16/MB1, both conditions."""
import gc
import tempfile
from time import perf_counter
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
from head_site import ActivationCapture
from head_site import parse_pressure_config
from initialization import apply_pythia_14m_initialization,_BASE
from optimizer_boundary import run_recipe_boundary,DynamicLossScaler,build_recipe_adamw
from run_config import (RUN_DIR,load_config,condition_specs,resolved_condition_config,
    parameter_sha256,EXPECTED_INITIAL_PARAMETER_SHA256,write_json,run_code_identity)
from sparsity_research.pythia import apply_activation_topology
from head_site import install_head,topology_metadata,load_checkpoint_pythia

if __name__=='__main__':
    torch.set_num_threads(1)
    c=load_config();rows=[];start=perf_counter()
    for condition in condition_specs(c):
        resolved=resolved_condition_config(c,condition)
        architecture=GPTNeoXConfig(**_BASE.EXPECTED_ARCHITECTURE)
        architecture.topology_id='HZ';architecture.site_gate=None
        architecture.site_gates=resolved['model']['site_gates']
        model=install_head(apply_activation_topology(GPTNeoXForCausalLM(architecture),torch=torch), resolved['model']['head_input_gate'])
        initialization=apply_pythia_14m_initialization(model,torch=torch)
        initial=parameter_sha256(model)
        if initial!=EXPECTED_INITIAL_PARAMETER_SHA256:raise RuntimeError('Initialization drift')
        optimizer,_=build_recipe_adamw(model,c['training'],torch=torch)
        pressure=parse_pressure_config(resolved['activation_pressure'])
        with ActivationCapture(model,list(pressure.sites),torch=torch) as capture:
            row=run_recipe_boundary(model=model,optimizer=optimizer,
                batches=[torch.arange(16).reshape(1,16)],pressure=pressure,capture=capture,
                loss_scaler=DynamicLossScaler(scale=1),gradient_clip_norm=1,
                torch=torch,device=torch.device('cpu'))
        if row['optimizer_step_skipped'] or not row['ol1_correction_applied']:
            raise RuntimeError('CPU mechanics failed')
        model.eval()
        ids=torch.arange(16).reshape(1,16)
        with torch.no_grad():
            logits=model(input_ids=ids,use_cache=False).logits
            changed=ids.clone();changed[:,5:]=changed[:,5:].flip(1)
            if not torch.equal(logits[:,:5],model(input_ids=changed,use_cache=False).logits[:,:5]):
                raise RuntimeError('Full-model prefix causality failed')
            with tempfile.TemporaryDirectory(prefix='run068-smoke-') as checkpoint:
                model.save_pretrained(checkpoint,safe_serialization=True)
                restored=load_checkpoint_pythia(GPTNeoXForCausalLM,checkpoint,torch=torch).eval()
                if parameter_sha256(restored)!=parameter_sha256(model) or topology_metadata(restored)!=topology_metadata(model):
                    raise RuntimeError('Full-model checkpoint identity differs')
                if not torch.equal(logits,restored(input_ids=ids,use_cache=False).logits):
                    raise RuntimeError('Full-model checkpoint logits differ')
                del restored
        local_checkpoint=RUN_DIR/'prelaunch/local-checkpoints'/condition['id']
        model.save_pretrained(local_checkpoint,safe_serialization=True)
        rows.append({'condition':condition['id'],'initial_parameter_sha256':initial,
                     'topology':topology_metadata(model),'boundary':row,
                     'checkpoint_logits_exact':True,'prefix_logits_exact':True})
        del model,optimizer,capture;gc.collect()
    write_json(RUN_DIR/'prelaunch/local-smoke.json',{'kind':'non_evidence_cpu_mechanics',
        'status':'passed','model':'real 14M architecture with approved random replay',
        'sequence_length':16,'micro_batch_size':1,'accumulation_steps':1,'precision':'float32',
        'device':'cpu','torch':torch.__version__,'rows':rows,'seconds':perf_counter()-start,
        'run_code':run_code_identity(),'limitation':'Does not establish CUDA FP16 MB32/GAS32 T2048 memory fit or ETC'})
