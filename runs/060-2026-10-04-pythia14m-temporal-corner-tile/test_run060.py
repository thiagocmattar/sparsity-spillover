"""Corner semantics, causality, exact ports, gradients and artifact contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_config import (load_config, condition_specs, resolved_condition_config,
    parameter_sha256, validate_config, build_schedule, REPO_ROOT,
    EXPECTED_SCHEDULE_SHA256, require_validation_coverage)
from tile_gate import CornerTileThreshold
from sparsity_research.pythia import apply_activation_topology, load_checkpoint_pythia, topology_metadata
from sparsity_research.capture import ActivationCapture
from sparsity_research.pressure import parse_pressure_config
from optimizer_boundary import run_recipe_boundary, DynamicLossScaler, build_recipe_adamw
from diagnostics import (SignEnergyAccumulator, CornerAccumulator, OccupancyAccumulator,
    GateInputCapture, activation_diagnostic_validation)

torch.set_num_threads(1)
CONFIG = load_config()
CONDITIONS = condition_specs(CONFIG)

def tiny(condition):
    torch.manual_seed(1234)
    cfg = GPTNeoXConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=2, max_position_embeddings=32,
        hidden_dropout=0, attention_dropout=0, use_cache=False)
    cfg._attn_implementation = 'eager'
    cfg.topology_id = 'HZ'; cfg.site_gate = None
    cfg.site_gates = resolved_condition_config(CONFIG, condition)['model']['site_gates']
    return apply_activation_topology(GPTNeoXForCausalLM(cfg), torch=torch)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('kappa', [0., .1])
def test_only_corner_controls_complete_tile_and_backward(dtype, kappa):
    x = torch.full((2, 16, 32), -2., dtype=dtype)
    x[:, ::8, ::16] = 1
    x[0, 0, 0] = kappa # Equality rejects in the comparison dtype.
    x[1, 8, 16] = 0
    x.requires_grad_()
    gate = CornerTileThreshold(kappa)
    expected = x.detach().clone()
    expected[0, :8, :16] = 0
    expected[1, 8:, 16:] = 0
    y = gate(x)
    assert torch.equal(y, expected)
    # A negative noncorner survives, even on the anchor row.
    assert y[0, 0, 17] == -2
    y.sum().backward()
    gradients = torch.ones_like(x)
    gradients[0, :8, :16] = 0; gradients[1, 8:, 16:] = 0
    assert torch.equal(x.grad, gradients)
    acc = SignEnergyAccumulator()
    acc.update('h.layer_0', x, y, gate.metadata(), torch=torch)
    assert acc.rows()[0]['rejected_count'] == 256
    corners = CornerAccumulator()
    corners.update('h.layer_0', x, y, gate.metadata(), torch=torch)
    assert corners.rows()[0]['tiles'] == 8 and corners.rows()[0]['empty'] == 2

@pytest.mark.parametrize('prefix', [1, 7, 8, 9, 15])
def test_gate_prefix_and_future_feature_invariance(prefix):
    torch.manual_seed(24)
    x = torch.randn(2, 16, 32)
    gate = CornerTileThreshold(.1)
    y = gate(x)
    altered = x.clone(); altered[:, prefix:] = torch.randn_like(altered[:, prefix:]) * 100
    assert torch.equal(y[:, :prefix], gate(altered)[:, :prefix])
    assert torch.equal(y[:, :prefix], gate(x[:, :prefix]))
    # All noncorner entries can change without changing tile decisions.
    altered = torch.full_like(x, -100)
    altered[:, ::8, ::16] = x[:, ::8, ::16]
    assert torch.equal(gate.retained_mask(x), gate.retained_mask(altered))

@pytest.mark.parametrize('condition', CONDITIONS, ids=[r['id'] for r in CONDITIONS])
def test_model_causality_ports_pressure_and_cache_guard(condition):
    model = tiny(condition).eval()
    batch = torch.arange(32).reshape(2, 16) % 64
    inputs = {}
    handles = []
    for i, layer in enumerate(model.gpt_neox.layers):
        for site, module in [('h',layer.mlp.dense_4h_to_h), ('z',layer.attention.dense)]:
            handles.append(module.register_forward_pre_hook(lambda _m, args, name=f'{site}.layer_{i}': inputs.update({name:args[0].detach().clone()})))
    with ActivationCapture(model,['a','m','h','z'],torch=torch) as capture, GateInputCapture(model) as before:
        original = model(input_ids=batch, use_cache=False).logits.detach()
        for name, value in inputs.items():
            assert torch.equal(value, capture.activations[name])
        for name, value in before.activations.items():
            site = name.split('.')[0]
            if site in ('h','z'):
                gate = CornerTileThreshold(**{k:v for k,v in model.config.site_gates[site].items() if k!='operator'})
                assert torch.equal(gate(value), capture.activations[name])
            else:
                assert torch.equal(value, capture.activations[name])
        expected_pressure = sum(capture.activations[f'h.layer_{i}'].float().abs().mean().item() for i in range(6))/6
    for handle in handles: handle.remove()
    changed = batch.clone(); changed[:, 5:] = torch.flip(changed[:, 5:], dims=[1])
    future = model(input_ids=changed, use_cache=False).logits.detach()
    assert torch.equal(original[:, :5], future[:, :5])
    with pytest.raises(ValueError,match='persistent tile state'):
        model(input_ids=batch, use_cache=True)
    optimizer,_ = build_recipe_adamw(model,CONFIG['training'],torch=torch)
    pressure = parse_pressure_config(resolved_condition_config(CONFIG,condition)['activation_pressure'])
    with ActivationCapture(model,['h'],torch=torch) as capture:
        row = run_recipe_boundary(model=model, optimizer=optimizer, batches=[batch], pressure=pressure,
            capture=capture, loss_scaler=DynamicLossScaler(scale=1), gradient_clip_norm=1,
            torch=torch, device=torch.device('cpu'))
    assert row['pressure_capture_names'] == sorted(f'h.layer_{i}' for i in range(6))
    assert row['pressure_capture_tensor_count'] == 6 and row['ol1_correction_applied']
    assert row['pressure_loss'] == pytest.approx(expected_pressure,rel=1e-6,abs=1e-8)
    assert not row['optimizer_step_skipped'] and row['pressure_to_task_ratio_final'] <= 1+1e-8

@pytest.mark.parametrize('condition', CONDITIONS, ids=[r['id'] for r in CONDITIONS])
def test_checkpoint_and_complete_diagnostics(condition, tmp_path):
    model = tiny(condition).eval()
    ids = torch.arange(32).reshape(1,32) % 64
    logits = model(input_ids=ids, use_cache=False).logits.detach()
    model.save_pretrained(tmp_path,safe_serialization=True)
    restored = load_checkpoint_pythia(GPTNeoXForCausalLM,tmp_path,torch=torch).eval()
    restored.set_attn_implementation('eager')
    assert parameter_sha256(restored) == parameter_sha256(model)
    assert topology_metadata(restored) == topology_metadata(model)
    assert torch.equal(restored(input_ids=ids,use_cache=False).logits,logits)
    cfg = deepcopy(CONFIG); cfg['data']['sequence_length'] = 32
    cfg['validation'].update(complete_sequences=3,input_tokens=96,excluded_tail_tokens=3,batch_size=2)
    coverage, stats = activation_diagnostic_validation(model=restored,tokens=np.arange(99,dtype=np.int32)%64,config=cfg,torch=torch,np=np)
    assert (coverage['sequences'],coverage['input_tokens'],coverage['excluded_tail_tokens']) == (3,96,3)
    assert len(stats['rows']) == 72 and len(stats['corner_rows']) == 12
    assert len(stats['occupancy_rows']) == 24 and len(stats['sign_energy_rows']) == 12
    assert all(r['rejected'] == r['empty'] and r['certificate_mismatches'] == 0 for r in stats['corner_rows'])
    json.loads(json.dumps(stats,allow_nan=False))

def test_integer_pooling_and_corrupt_certificate():
    gate = CornerTileThreshold(.1); acc = CornerAccumulator(); occupancy = OccupancyAccumulator([(8,16)])
    x = torch.ones(1,8,32); x[0,0,0] = 0
    acc.update('h.layer_0',x,gate(x),gate.metadata(),torch=torch)
    y = torch.ones(2,8,32)
    acc.update('h.layer_1',y,gate(y),gate.metadata(),torch=torch)
    assert acc.pooled()[0]['tiles'] == 6 and acc.pooled()[0]['empty'] == 1
    for name,value in [('h.layer_0',gate(x)),('h.layer_1',gate(y))]: occupancy.update(name,value,torch=torch)
    assert occupancy.pooled()[0]['tiles']['8x16']['empty_fraction'] == pytest.approx(1/6)
    corrupted = gate(x); corrupted[0,1,1] = 1
    with pytest.raises(RuntimeError,match='certify'):
        acc.update('h.layer_2',x,corrupted,gate.metadata(),torch=torch)

def test_recipe_schedule_coverage_and_invalid_geometry():
    assert len(CONDITIONS) == 2
    for c in CONDITIONS:
        assert c['pressure_sites'] == ['h'] and c['active_sites'] == ['h','z']
    meta=json.loads((REPO_ROOT/CONFIG['data']['training_metadata']).read_text())
    _, digest, summary = build_schedule(CONFIG,meta,np=np)
    assert digest == EXPECTED_SCHEDULE_SHA256 and summary['wrapped_blocks'] == 714
    for k in (-1,float('nan'),True):
        cfg=deepcopy(CONFIG); cfg['conditions']['gate_thresholds']=[k]
        with pytest.raises(ValueError): validate_config(cfg)
    for r in (0,True,3):
        cfg=deepcopy(CONFIG); cfg['conditions']['tile_tokens']=r
        with pytest.raises(ValueError): validate_config(cfg)
    with pytest.raises(ValueError): CornerTileThreshold(.1)(torch.ones(1,8,17))
    with pytest.raises(RuntimeError):
        require_validation_coverage({'sequences':337,'input_tokens':692224,'excluded_tail_tokens':1444,'complete_block_coverage':True},CONFIG)

def test_training_recovery_serializes_optimizer_scaler_rng_and_gate(tmp_path):
    from training import _save_checkpoint
    model = tiny(CONDITIONS[0])
    optimizer,_ = build_recipe_adamw(model, CONFIG['training'],torch=torch)
    scaler = DynamicLossScaler(scale=1)
    pressure = parse_pressure_config(resolved_condition_config(CONFIG,CONDITIONS[0])['activation_pressure'])
    with ActivationCapture(model,['h'],torch=torch) as capture:
        run_recipe_boundary(model=model,optimizer=optimizer,batches=[torch.arange(16).reshape(1,16)],
            pressure=pressure,capture=capture,loss_scaler=scaler,gradient_clip_norm=1,
            torch=torch,device=torch.device('cpu'))
    checkpoint = _save_checkpoint(model=model,optimizer=optimizer,scaler=scaler,step=1,
        include_optimizer=True,root=tmp_path,schedule_hash='test',torch=torch)
    state = torch.load(checkpoint/'training_state.pt',weights_only=True)
    assert state['optimizer'] and state['optimizer_saved'] and state['step'] == 1
    assert state['loss_scaler'] == scaler.state_dict() and state['schedule_hash'] == 'test'
    assert torch.equal(state['torch_rng_state'],torch.get_rng_state())
    assert state['cuda_rng_states'] == []
    restored = load_checkpoint_pythia(GPTNeoXForCausalLM,checkpoint,torch=torch)
    assert parameter_sha256(restored) == parameter_sha256(model)
    assert topology_metadata(restored) == topology_metadata(model)

def test_logical_counter_uses_post_tile_operands_and_same_hz_ceiling():
    from sparsity_research.logical_capture import capture_logical_products, LogicalProductAccumulator
    from sparsity_research.metrics import LOGICAL_OPERATIONS
    from sparsity_research.ceilings import architecture_ceiling
    model = tiny(CONDITIONS[0]).eval()
    acc = LogicalProductAccumulator()
    with capture_logical_products(model,accumulator=acc,torch=torch):
        model(input_ids=torch.arange(16).reshape(1,16),use_cache=False)
    measured = acc.summary(model=model,total_input_tokens=16)
    assert set(measured['per_operation']) == set(LOGICAL_OPERATIONS)
    maximum = architecture_ceiling('HZ',layers=6,hidden_size=16,ffn_size=32,sequence_length=16,vocabulary_size=64)
    assert maximum
