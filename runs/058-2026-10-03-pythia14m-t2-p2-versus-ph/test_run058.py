"""Mathematics, actual six-layer hooks, OL1 capture, serialization and coverage."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
from sparsity_research.capture import ActivationCapture
from sparsity_research.pressure import activation_l1,parse_pressure_config
from sparsity_research.pythia import apply_activation_topology,load_checkpoint_pythia,topology_metadata
from sparsity_research.sites import FixedOneSidedThreshold,FixedSymmetricThreshold

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_config import load_config,condition_specs,resolved_condition_config,parameter_sha256,validate_config,build_schedule
from optimizer_boundary import run_recipe_boundary,DynamicLossScaler,build_recipe_adamw
from diagnostics import GateInputCapture,SignEnergyAccumulator,OccupancyAccumulator,activation_diagnostic_validation

torch.set_num_threads(1)
CONFIG=load_config()
CONDITIONS=condition_specs(CONFIG)

def tiny(condition):
    torch.manual_seed(1234)
    config=GPTNeoXConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=6,num_attention_heads=2,max_position_embeddings=32,
        hidden_dropout=0,attention_dropout=0,use_cache=False)
    config._attn_implementation='eager'
    config.topology_id='HZ'; config.site_gate=None
    config.site_gates=resolved_condition_config(CONFIG,condition)['model']['site_gates']
    return apply_activation_topology(GPTNeoXForCausalLM(config),torch=torch)

@pytest.mark.parametrize('operator',[FixedOneSidedThreshold,FixedSymmetricThreshold])
def test_threshold_equality_and_backward(operator):
    x=torch.tensor([-2.,-.1,-.05,0.,.05,.1,2.],requires_grad=True)
    y=operator(.1)(x)
    expected=torch.tensor([0.,0.,0.,0.,0.,.1,2.]) if operator is FixedOneSidedThreshold else torch.tensor([-2.,-.1,0.,0.,0.,.1,2.])
    assert torch.equal(y,expected)
    y.sum().backward()
    assert torch.equal(x.grad, (x.detach()>=.1 if operator is FixedOneSidedThreshold else x.detach().abs()>=.1).float())
    assert torch.equal(FixedSymmetricThreshold(0)(x),x)

@pytest.mark.parametrize('condition',CONDITIONS,ids=[r['id'] for r in CONDITIONS])
def test_exact_hooks_and_boundary(condition):
    model=tiny(condition)
    # Nontrivial affine params ensure gate input is AFTER gamma/beta, not before.
    with torch.no_grad():
        model.gpt_neox.layers[0].input_layernorm.weight.fill_(1.7)
        model.gpt_neox.layers[0].input_layernorm.bias.fill_(-.3)
    ln_raw=[];qkv_inputs=[];w1_inputs=[];wo_inputs=[];w2_inputs=[]
    layer=model.gpt_neox.layers[0]
    handles=[layer.input_layernorm.register_forward_pre_hook(lambda m,i:ln_raw.append(i[0].detach().clone())),
        layer.attention.query_key_value.register_forward_pre_hook(lambda m,i:qkv_inputs.append(i[0].detach().clone())),
        layer.mlp.dense_h_to_4h.register_forward_pre_hook(lambda m,i:w1_inputs.append(i[0].detach().clone())),
        layer.attention.dense.register_forward_pre_hook(lambda m,i:wo_inputs.append(i[0].detach().clone())),
        layer.mlp.dense_4h_to_h.register_forward_pre_hook(lambda m,i:w2_inputs.append(i[0].detach().clone()))]
    batch=torch.arange(32).reshape(2,16)%64
    pressure=parse_pressure_config(resolved_condition_config(CONFIG,condition)['activation_pressure'])
    with ActivationCapture(model,['a','m','h','z'],torch=torch) as all_sites,GateInputCapture(model) as before:
        model(input_ids=batch,labels=batch)
        expected=torch.nn.functional.layer_norm(ln_raw[0],layer.input_layernorm.normalized_shape,
            layer.input_layernorm.weight,layer.input_layernorm.bias,layer.input_layernorm.eps)
        assert torch.equal(expected,before.activations['a.layer_0'])
        for value,name in ((qkv_inputs[0],'a'),(w1_inputs[0],'m'),(wo_inputs[0],'z'),(w2_inputs[0],'h')):
            assert torch.equal(value,all_sites.activations[f'{name}.layer_0'])
        expected_mean=sum(all_sites.activations[f'{s}.layer_{i}'].float().abs().mean().item()
                          for s in pressure.sites for i in range(6))/(len(pressure.sites)*6)
    for handle in handles:handle.remove()
    optimizer,_=build_recipe_adamw(model,CONFIG['training'],torch=torch)
    with ActivationCapture(model,list(pressure.sites),torch=torch) as capture:
        result=run_recipe_boundary(model=model,optimizer=optimizer,batches=[batch],pressure=pressure,
            capture=capture,loss_scaler=DynamicLossScaler(scale=1),gradient_clip_norm=1,
            torch=torch,device=torch.device('cpu'))
    assert result['pressure_loss']==pytest.approx(expected_mean,rel=1e-6)
    assert result['pressure_capture_tensor_count']==len(pressure.sites)*6
    assert set(result['pressure_capture_names'])=={f'{s}.layer_{i}' for s in pressure.sites for i in range(6)}
    assert not result['optimizer_step_skipped'] and result['ol1_correction_applied']
    assert result['pressure_to_task_ratio_final']<=1+1e-8
    assert result['gradient_clipping_site']=='task_gradients_before_adamw'
    assert result['pressure_gradient_norm']>0

def test_capture_mismatch_fails_before_update():
    condition=CONDITIONS[-1];model=tiny(condition)
    optimizer,_=build_recipe_adamw(model,CONFIG['training'],torch=torch)
    with ActivationCapture(model,['h'],torch=torch) as capture:
        with pytest.raises(RuntimeError,match='capture mismatch'):
            run_recipe_boundary(model=model,optimizer=optimizer,batches=[torch.ones((1,16),dtype=torch.long)],
                pressure=parse_pressure_config(resolved_condition_config(CONFIG,condition)['activation_pressure']),
                capture=capture,loss_scaler=DynamicLossScaler(scale=1),gradient_clip_norm=1,
                torch=torch,device=torch.device('cpu'))
    assert not optimizer.state

def test_signed_energy_counts_pool_before_dividing():
    acc=SignEnergyAccumulator()
    x=torch.tensor([-2.,-.1,0.,.05,.1,2.])
    spec={'operator':'one_sided_threshold','kappa':.1}
    acc.update('a.layer_0',x,FixedOneSidedThreshold(.1)(x),spec,torch=torch)
    acc.update('a.layer_1',x[:1],FixedOneSidedThreshold(.1)(x[:1]),spec,torch=torch)
    row=acc.pooled()[0]
    assert row['elements']==7 and row['pre_negative']==3 and row['post_negative']==0
    assert row['rejected_large_negative_count']==3
    assert row['rejected_square_sum']==pytest.approx(8.0125)
    acc=SignEnergyAccumulator();spec['operator']='symmetric_threshold'
    acc.update('a.layer_0',x,FixedSymmetricThreshold(.1)(x),spec,torch=torch)
    assert acc.rows()[0]['rejected_large_negative_count']==0

def test_sequence_aligned_tiles_and_row_histogram():
    x=torch.zeros((2,16,32));x[0,0,0]=1;x[1,15,31]=1
    acc=OccupancyAccumulator();acc.update('m.layer_0',x,torch=torch)
    row=acc.pooled()[0]
    assert row['rows']==32 and row['row_nonzero_histogram'][0]==30
    assert row['tiles']['8x16']=={'empty':6,'total':8,'empty_fraction':.75}
    assert row['tiles']['16x16']=={'empty':2,'total':4,'empty_fraction':.5}
    # An unequal second layer checks pooling counters, not averaging rates.
    acc.update('m.layer_1',torch.zeros((1,16,32)),torch=torch)
    assert acc.pooled()[0]['tiles']['8x16']['empty_fraction']==pytest.approx(10/12)

@pytest.mark.parametrize('operator',[FixedOneSidedThreshold,FixedSymmetricThreshold])
def test_energy_audit_respects_fp16_threshold_equality(operator):
    x=torch.tensor([-.1,.1],dtype=torch.float16)
    acc=SignEnergyAccumulator()
    spec={'operator':'one_sided_threshold' if operator is FixedOneSidedThreshold else 'symmetric_threshold','kappa':.1}
    acc.update('h.layer_0',x,operator(.1)(x),spec,torch=torch)
    assert acc.rows()[0]['effective_kappa']==float(x[1])
    assert acc.rows()[0]['post_positive']==1

@pytest.mark.parametrize('index',[0,1,2,3])
def test_checkpoint_roundtrip_and_full_diagnostic_pipeline(index,tmp_path):
    model=tiny(CONDITIONS[index]);model.eval()
    before=parameter_sha256(model);model.save_pretrained(tmp_path,safe_serialization=True)
    restored=load_checkpoint_pythia(GPTNeoXForCausalLM,tmp_path,torch=torch)
    assert parameter_sha256(restored)==before
    assert topology_metadata(restored)==topology_metadata(model)
    cfg=deepcopy(CONFIG)
    cfg['data']['sequence_length']=16
    cfg['validation'].update(complete_sequences=3,input_tokens=48,excluded_tail_tokens=3,batch_size=2)
    tokens=np.arange(51,dtype=np.int32)%64
    coverage,stats=activation_diagnostic_validation(model=restored,tokens=tokens,config=cfg,torch=torch,np=np)
    assert (coverage['sequences'],coverage['input_tokens'],coverage['excluded_tail_tokens'])==(3,48,3)
    assert len(stats['rows'])==60 and len(stats['occupancy_rows'])==24
    assert len(stats['sign_energy_rows'])==12 and len(stats['layernorm_affine'])==12
    assert sum(r['total'] for r in stats['rows'] if r['name'].startswith('a_pre.'))==6*48*16
    json.loads(json.dumps(stats,allow_nan=False))

def test_actual_cache_and_schedule_identity():
    from run_config import REPO_ROOT,EXPECTED_SCHEDULE_SHA256,require_validation_coverage
    meta=json.loads((REPO_ROOT/CONFIG['data']['training_metadata']).read_text())
    _,digest,summary=build_schedule(CONFIG,meta,np=np)
    assert digest==EXPECTED_SCHEDULE_SHA256
    assert summary['scheduled_blocks']==729088 and summary['wrapped_blocks']==714
    with pytest.raises(RuntimeError,match='coverage'):
        require_validation_coverage({'sequences':337,'input_tokens':692224,'excluded_tail_tokens':1444,'complete_block_coverage':True},CONFIG)

def test_invalid_threshold_and_batch():
    for value in (-1,float('nan'),True):
        cfg=deepcopy(CONFIG);cfg['conditions']['gate_thresholds']=[value]
        with pytest.raises(ValueError):validate_config(cfg)
    cfg=deepcopy(CONFIG);cfg['training']['micro_batch_size']=31
    with pytest.raises(ValueError,match='decomposition'):validate_config(cfg)

def test_training_checkpoint_retains_optimizer_scaler_rng_and_topology(tmp_path):
    from training import _save_checkpoint
    model=tiny(CONDITIONS[-1])
    optimizer,_=build_recipe_adamw(model,CONFIG['training'],torch=torch)
    scaler=DynamicLossScaler(scale=1)
    pressure=parse_pressure_config(resolved_condition_config(CONFIG,CONDITIONS[-1])['activation_pressure'])
    with ActivationCapture(model,list(pressure.sites),torch=torch) as capture:
        run_recipe_boundary(model=model,optimizer=optimizer,batches=[torch.arange(16).reshape(1,16)],
            pressure=pressure,capture=capture,loss_scaler=scaler,gradient_clip_norm=1,
            torch=torch,device=torch.device('cpu'))
    target=_save_checkpoint(model=model,optimizer=optimizer,scaler=scaler,step=1,
        include_optimizer=True,root=tmp_path,schedule_hash='test',torch=torch)
    state=torch.load(target/'training_state.pt',weights_only=True)
    assert state['optimizer'] and state['optimizer_saved'] and state['step']==1
    assert state['loss_scaler']==scaler.state_dict() and state['schedule_hash']=='test'
    assert torch.equal(state['torch_rng_state'],torch.get_rng_state())
    assert state['cuda_rng_states']==[]
    restored=load_checkpoint_pythia(GPTNeoXForCausalLM,target,torch=torch)
    assert parameter_sha256(restored)==parameter_sha256(model)
    assert topology_metadata(restored)==topology_metadata(model)


def test_t2_has_no_am_gate_or_pressure_and_ph_p2_are_matched():
    assert len(CONDITIONS)==4
    for condition in CONDITIONS:
        resolved=resolved_condition_config(CONFIG,condition)
        model=tiny(condition)
        assert set(resolved['model']['site_gates'])=={'h','z'}
        assert not hasattr(model.gpt_neox.layers[0],'a_gate')
        assert not hasattr(model.gpt_neox.layers[0],'m_gate')
        assert condition['pressure_sites']==(['h'] if condition['pressure_recipe']=='ph' else ['h','z'])
        assert condition['gate_operator']=='one_sided_threshold'
        assert topology_metadata(model)['active_sites']==['h','z']
