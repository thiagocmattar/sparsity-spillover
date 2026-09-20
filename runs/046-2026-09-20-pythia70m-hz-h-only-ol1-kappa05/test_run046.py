import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@pytest.fixture(scope='module')
def modules():
    names = ['run_config','training','initialization','initialization_artifact','model_factory',
             'optimizer_boundary','diagnostics','verification','_reuse_run004','_reuse_run017',
             '_reuse_run018','run046_capture','run046_topology']
    previous = {n:sys.modules.pop(n,None) for n in names}
    sys.path.insert(0,str(HERE))
    try:
        yield importlib.import_module('run_config'), importlib.import_module('training')
    finally:
        sys.path.remove(str(HERE))
        for n,v in previous.items():
            sys.modules.pop(n,None)
            if v is not None:sys.modules[n]=v


def test_matched_70m_recipe_and_one_hz_condition(modules):
    rc,_=modules;c=rc.load_config()
    prior=yaml.safe_load((ROOT/'runs/034-2026-09-17-pythia70m-h-only-ol1/config.yaml').read_text())
    for k in ['initialization_artifact','recipe','runtime','data','seeds','training','validation','checkpoints','artifacts']:
        assert c[k]==prior[k],k
    assert {k:v for k,v in c['model'].items() if k not in ['topology_id','site_gate','site_gates']}==prior['model']
    assert {k:v for k,v in c['diagnostics'].items() if k!='clipping_frontier'}==prior['diagnostics']
    assert c['diagnostics']['clipping_frontier'] is None
    rows=rc.condition_specs(c)
    assert len(rows)==1 and [r['gate_threshold'] for r in rows]==[.5]
    for r in rows:
        resolved=rc.resolved_condition_config(c,r)
        assert resolved['model']['site_gates']==rc.site_gates(r['gate_threshold'])
        assert set(resolved['model']['site_gates'])=={'h','z'}
        assert resolved['activation_pressure']==dict(method='orthogonal_l1',sites=['h'],weight=1.,step_budget=1.,eps=1e-12)
        assert rc.worker_conditions(c,r['id'])==[r]
    metadata=json.loads((ROOT/c['data']['training_metadata']).read_text())
    assert rc.build_schedule(c,metadata,np=np)[1]==rc.EXPECTED_SCHEDULE_SHA256


def test_real_70m_random_identity_pressure_and_gate_hooks(modules,tmp_path):
    rc,tr=modules
    from model_factory import build_pinned_run046_model
    from initialization_artifact import load_pinned_initialization
    from transformers import AutoModelForCausalLM
    from sparsity_research.capture import ActivationCapture
    from sparsity_research.pressure import activation_l1
    from sparsity_research.pythia import topology_metadata,load_checkpoint_pythia
    c=rc.load_config();resolved=rc.resolved_condition_config(c,rc.condition_specs(c)[0])
    model=build_pinned_run046_model(resolved['model'],device=torch.device('cpu'),torch=torch,auto_model=AutoModelForCausalLM)
    load_pinned_initialization(model,torch=torch)
    assert rc.parameter_sha256(model)==rc.EXPECTED_INITIAL_PARAMETER_SHA256
    assert topology_metadata(model)['active_sites']==['h','z']
    with tr.HOnlyPressureCapture(model,['h'],torch=torch) as pressure, ActivationCapture(model,['h','z'],torch=torch) as gates, torch.no_grad():
        logits=model(input_ids=torch.tensor([[1,2,3,4]])).logits
        assert tuple(sorted(pressure.activations))==rc.EXPECTED_PRESSURE_CAPTURE_NAMES
        assert all(t.shape==(1,4,2048) for t in pressure.activations.values())
        assert len(gates.activations)==12
        assert all(torch.all(t[t!=0]>=.5) for t in gates.activations.values())
        assert torch.equal(activation_l1(pressure.activations),torch.stack([t.abs().mean() for n,t in gates.activations.items() if n.startswith('h.')]).mean())
    for layer in model.gpt_neox.layers:
        assert not hasattr(layer,'a_gate') and not hasattr(layer,'m_gate')
        assert layer.mlp.act.kappa==.5 and layer.attention.z_gate.kappa==.5
    model.save_pretrained(tmp_path)
    loaded=load_checkpoint_pythia(AutoModelForCausalLM,tmp_path,torch=torch)
    with torch.no_grad():assert torch.equal(logits,loaded(input_ids=torch.tensor([[1,2,3,4]])).logits)


def test_hz_integer_ceiling_and_verified_comparator(modules):
    rc,_=modules
    from sparsity_research.ceilings import architecture_ceiling
    from verification import _load_run034_verification
    ceiling=architecture_ceiling('HZ',layers=6,hidden_size=512,ffn_size=2048,sequence_length=2048,vocabulary_size=50304)
    assert ceiling['reachable_product_count']==rc.EXPECTED_CEILING_NUMERATOR
    assert ceiling['model_product_count']==rc.EXPECTED_CEILING_DENOMINATOR
    assert _load_run034_verification()['condition_count']==10


def test_h_gate_boundary_equality_and_gradient(modules):
    from sparsity_research.sites import FixedOneSidedThreshold
    x=torch.tensor([-.2,0.,.499,.5,.8],requires_grad=True)
    y=FixedOneSidedThreshold(.5)(x);y.sum().backward()
    assert torch.equal(y,torch.tensor([0.,0.,0.,.5,.8]))
    assert torch.equal(x.grad,torch.tensor([0.,0.,0.,1.,1.]))
