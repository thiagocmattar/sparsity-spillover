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
from sparsity_research.pythia import apply_activation_topology
from head_site import install_head, load_checkpoint_pythia, topology_metadata, ActivationCapture, parse_pressure_config, HEAD_NAME
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
    return install_head(apply_activation_topology(GPTNeoXForCausalLM(cfg), torch=torch), resolved_condition_config(CONFIG,condition)['model']['head_input_gate'])

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
    with ActivationCapture(model,['a','m','h','z','head_in'],torch=torch) as capture, GateInputCapture(model) as before:
        original = model(input_ids=batch, use_cache=False).logits.detach()
        for name, value in inputs.items():
            assert torch.equal(value, capture.activations[name])
        for name, value in before.activations.items():
            site = name.split('.')[0]
            if site in ('h','z','head_in'):
                gate = CornerTileThreshold(**{k:v for k,v in (model.config.head_input_gate if site=='head_in' else model.config.site_gates[site]).items() if k!='operator'})
                assert torch.equal(gate(value), capture.activations[name])
            else:
                assert torch.equal(value, capture.activations[name])
        names = sorted([f'h.layer_{i}' for i in range(6)]+([HEAD_NAME] if 'head_in' in condition['pressure_sites'] else []))
        expected_pressure = sum(capture.activations[name].float().abs().mean().item() for name in names)/len(names)
    for handle in handles: handle.remove()
    changed = batch.clone(); changed[:, 5:] = torch.flip(changed[:, 5:], dims=[1])
    future = model(input_ids=changed, use_cache=False).logits.detach()
    assert torch.equal(original[:, :5], future[:, :5])
    with pytest.raises(ValueError,match='persistent tile state'):
        model(input_ids=batch, use_cache=True)
    optimizer,_ = build_recipe_adamw(model,CONFIG['training'],torch=torch)
    pressure = parse_pressure_config(resolved_condition_config(CONFIG,condition)['activation_pressure'])
    with ActivationCapture(model,condition['pressure_sites'],torch=torch) as capture:
        row = run_recipe_boundary(model=model, optimizer=optimizer, batches=[batch], pressure=pressure,
            capture=capture, loss_scaler=DynamicLossScaler(scale=1), gradient_clip_norm=1,
            torch=torch, device=torch.device('cpu'))
    assert row['pressure_capture_names'] == names
    assert row['pressure_capture_tensor_count'] == len(names) and row['ol1_correction_applied']
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
    assert len(stats['rows']) == 74 and len(stats['corner_rows']) == 13
    assert len(stats['occupancy_rows']) == 25 and len(stats['sign_energy_rows']) == 13
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
    assert len(CONDITIONS) == 4
    for c in CONDITIONS:
        assert c['pressure_sites'] in (['h'],['h','head_in']) and c['active_sites'] == ['h','z','head_in']
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


@pytest.mark.parametrize('condition', CONDITIONS, ids=[r['id'] for r in CONDITIONS])
def test_head_port_after_affine_normalization_and_dense_logits(condition):
    model = tiny(condition).eval()
    ln = model.gpt_neox.final_layer_norm
    with torch.no_grad():
        ln.weight.copy_(torch.linspace(.5,2,16))
        ln.bias.copy_(torch.linspace(-.3,.3,16))
    residual, operand = [], []
    handles = [ln.register_forward_pre_hook(lambda _m,args: residual.append(args[0].detach())),
        model.embed_out.register_forward_pre_hook(lambda _m,args: operand.append(args[0].detach()))]
    ids = torch.arange(32).reshape(1,32)
    with GateInputCapture(model) as before, ActivationCapture(model,['head_in'],torch=torch) as capture:
        logits = model(input_ids=ids,use_cache=False).logits
        normalized = torch.nn.functional.layer_norm(residual[0],(16,),ln.weight,ln.bias,ln.eps)
        assert torch.equal(before.activations[HEAD_NAME],normalized)
        expected = model.head_input_gate(normalized)
        assert torch.equal(capture.activations[HEAD_NAME],expected)
        assert torch.equal(operand[0],expected)
        assert torch.equal(logits,torch.nn.functional.linear(expected,model.embed_out.weight))
    for handle in handles: handle.remove()


@pytest.mark.parametrize('condition', CONDITIONS, ids=[r['id'] for r in CONDITIONS])
def test_pressure_tensor_mean_and_head_gradient_scope(condition):
    from sparsity_research.pressure import activation_l1
    model = tiny(condition)
    sites = condition['pressure_sites']
    with ActivationCapture(model,sites,torch=torch) as capture:
        model(input_ids=torch.arange(32).reshape(1,32),use_cache=False)
        loss = activation_l1(capture.activations)
        terms = [capture.activations[f'h.layer_{i}'].float().abs().mean() for i in range(6)]
        if 'head_in' in sites: terms.append(capture.activations[HEAD_NAME].float().abs().mean())
        expected = torch.stack(terms).mean()
        assert torch.equal(loss,expected)
        parameters = [model.gpt_neox.final_layer_norm.weight,model.embed_out.weight]
        actual = torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
        manual = torch.autograd.grad(expected,parameters,allow_unused=True)
        assert actual[1] is None and manual[1] is None
        if 'head_in' in sites:
            assert torch.equal(actual[0],manual[0]) and torch.count_nonzero(actual[0]) > 0
        else:
            assert actual[0] is None and manual[0] is None


@pytest.mark.parametrize('fault',['missing','extra'])
def test_boundary_rejects_wrong_capture_before_optimizer_update(fault):
    condition = CONDITIONS[2]
    model = tiny(condition)
    optimizer,_ = build_recipe_adamw(model,CONFIG['training'],torch=torch)
    pressure = parse_pressure_config(resolved_condition_config(CONFIG,condition)['activation_pressure'])
    original = parameter_sha256(model)
    with ActivationCapture(model,condition['pressure_sites'],torch=torch) as capture:
        def corrupt(_m,_args,_output):
            if fault == 'missing': capture.activations.pop(HEAD_NAME)
            else: capture.activations['extra'] = capture.activations[HEAD_NAME]
        handle = model.register_forward_hook(corrupt)
        try:
            with pytest.raises(RuntimeError,match='capture mismatch'):
                run_recipe_boundary(model=model,optimizer=optimizer,batches=[torch.arange(16).reshape(1,16)],
                    pressure=pressure,capture=capture,loss_scaler=DynamicLossScaler(scale=1),
                    gradient_clip_norm=1,torch=torch,device=torch.device('cpu'))
        finally: handle.remove()
    assert parameter_sha256(model) == original and not optimizer.state


@pytest.mark.parametrize('gated',[True,False])
def test_head_pooled_elements_tiles_and_separate_logical_accounting(gated):
    from diagnostics import logical_product_validation
    model = tiny(CONDITIONS[0]).eval()
    if not gated: install_head(model,None)
    cfg = deepcopy(CONFIG); cfg['data']['sequence_length'] = 32
    cfg['validation'].update(complete_sequences=3,input_tokens=96,excluded_tail_tokens=3,batch_size=2)
    tokens = np.arange(99,dtype=np.int32)%64
    coverage,stats = activation_diagnostic_validation(model=model,tokens=tokens,config=cfg,torch=torch,np=np)
    head = next(r for r in stats['rows'] if r['name']==HEAD_NAME)
    pre = next(r for r in stats['rows'] if r['name']=='head_in_pre.final')
    occupancy = next(r for r in stats['occupancy_rows'] if r['name']==HEAD_NAME)
    assert head['total']==pre['total']==1536 and occupancy['rows']==96
    assert occupancy['tiles']['8x16']['total']==12
    assert sum(occupancy['row_nonzero_histogram'])==96
    assert len(stats['rows'])==74 and len(stats['layernorm_affine'])==13
    assert len(stats['corner_rows'])==(13 if gated else 12)
    if gated:
        corner=next(r for r in stats['corner_rows'] if r['name']==HEAD_NAME)
        assert corner['empty']==occupancy['tiles']['8x16']['empty']
        assert corner['tiles']==12 and sum(corner['position_zeros'])==head['exact_zero_count']
    else:
        assert head['exact_zero_count']==pre['exact_zero_count']
    logical_coverage,measured,maximum=logical_product_validation(model=model,tokens=tokens,config=cfg,
        torch=torch,np=np,activation_statistics=stats)
    assert logical_coverage['input_tokens']==coverage['input_tokens']
    assert measured['head_input_opportunity']['products']==1536*64
    assert measured['head_input_opportunity']['zero_activation_products']==head['exact_zero_count']*64
    assert measured['head_input_eager_opportunity']['products']==1536*64
    assert 'excludes' in maximum['scope']
    json.loads(json.dumps(measured,allow_nan=False))


def test_retained_control_inputs_and_identity_reload():
    import importlib.util
    spec=importlib.util.spec_from_file_location('run066_controls',Path(__file__).parent/'17_controls.py')
    controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
    inventory=controls.verify_inputs()
    for row in inventory['controls']:
        model=load_checkpoint_pythia(GPTNeoXForCausalLM,controls.RUN_DIR/row['checkpoint'],torch=torch).eval()
        model.set_attn_implementation('eager')
        assert parameter_sha256(model)==row['parameter_sha256']
        assert topology_metadata(model)['head_input_gate'] is None
        values=[]
        handle=model.head_input_gate.register_forward_hook(lambda _m,args,value:values.append(torch.equal(args[0],value)))
        model(input_ids=torch.arange(16).reshape(1,16),use_cache=False)
        handle.remove()
        assert values==[True]


def test_head_checkpoint_drift_is_rejected():
    model=tiny(CONDITIONS[0])
    model.config.head_input_gate['kappa']=.1
    with pytest.raises(ValueError,match='Realized head gate'): topology_metadata(model)


@pytest.mark.parametrize('gated',[False,True])
def test_production_width_diagnostic_verifier_rejects_bad_tile_count(gated):
    import importlib.util
    spec=importlib.util.spec_from_file_location('run066_verify',Path(__file__).parent/'03_verify.py')
    verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
    inventory=json.loads((Path(__file__).parent/'inputs/controls/inventory.json').read_text())
    model=load_checkpoint_pythia(GPTNeoXForCausalLM,Path(__file__).parent/inventory['controls'][0]['checkpoint'],torch=torch).eval()
    model.set_attn_implementation('eager')
    if gated: install_head(model,resolved_condition_config(CONFIG,CONDITIONS[0])['model']['head_input_gate'])
    cfg=deepcopy(CONFIG);cfg['data']['sequence_length']=32
    cfg['validation'].update(complete_sequences=3,input_tokens=96,excluded_tail_tokens=3,batch_size=2)
    _,stats=activation_diagnostic_validation(model=model,tokens=np.arange(99,dtype=np.int32),config=cfg,torch=torch,np=np)
    verify.verify_head_statistics(stats,cfg,gated=gated)
    broken=deepcopy(stats)
    head=next(r for r in broken['occupancy_rows'] if r['name']==HEAD_NAME)
    head['tiles']['8x16']['total']-=1
    with pytest.raises(ValueError,match='Head tile counts incomplete'):
        verify.verify_head_statistics(broken,cfg,gated=gated)
