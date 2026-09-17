import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from sparsity_research.pressure import parse_pressure_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@pytest.fixture(scope='module')
def modules():
    names = ['run_config', 'training', 'optimizer_boundary', 'initialization',
             'initialization_artifact', 'model_factory', 'diagnostics', 'verification',
             '_reuse_run004', '_reuse_run017', '_reuse_run018', 'run034_capture']
    previous = {name:sys.modules.pop(name, None) for name in names}
    sys.path.insert(0, str(HERE))
    try:
        import run_config, optimizer_boundary, training
        yield run_config, optimizer_boundary, None, training
    finally:
        sys.path.remove(str(HERE))
        for name, value in previous.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


def test_ten_conditions_match_source_recipe_and_gate_semantics(modules):
    cfg, _, _, _ = modules
    config = cfg.load_config()
    old = yaml.safe_load((ROOT/'runs/018-2026-09-01-pythia70m-selected-ladder-canonical-init/config.yaml').read_text())
    for key in ['model','initialization_artifact','recipe','runtime','data','seeds','training','validation','diagnostics']:
        assert config[key] == old[key], key
    rows = cfg.condition_specs(config)
    assert [r['id'] for r in rows] == list(cfg.EXPECTED_CONDITION_IDS)
    assert len(rows) == 10
    for row in rows:
        resolved = cfg.resolved_condition_config(config, row)
        sites = cfg.A4_SITES if row['id'].startswith('a4') else cfg.A7_SITES
        assert row['active_sites'] == list(sites)
        assert resolved['model']['site_gates'] == cfg.site_gates(row['topology_id'], row['gate_threshold'])
        assert resolved['activation_pressure'] == dict(method='orthogonal_l1',sites=['h'],weight=1.,step_budget=1.,eps=1e-12)
        assert cfg.worker_conditions(config, row['id']) == [row]
    metadata = json.loads((ROOT/config['data']['training_metadata']).read_text())
    _, digest, schedule = cfg.build_schedule(config, metadata, np=np)
    assert digest == cfg.EXPECTED_SCHEDULE_SHA256
    assert schedule['scheduled_blocks'] == 729088


@pytest.mark.parametrize('family', ['a4', 'a7'])
def test_real_70m_canonical_load_and_only_h_hooks(modules, family):
    cfg, _, _, _ = modules
    from initialization_artifact import load_pinned_initialization
    from model_factory import build_pinned_run034_model
    from run034_capture import HOnlyPressureCapture
    from transformers import AutoModelForCausalLM
    config = cfg.load_config()
    row = next(r for r in cfg.condition_specs(config) if r['id'] == family+'-h-ol1-kappa-0p5')
    resolved = cfg.resolved_condition_config(config,row)
    model = build_pinned_run034_model(resolved['model'],device=torch.device('cpu'),torch=torch,auto_model=AutoModelForCausalLM)
    load_pinned_initialization(model,torch=torch)
    assert cfg.parameter_sha256(model) == cfg.EXPECTED_INITIAL_PARAMETER_SHA256
    with HOnlyPressureCapture(model,['h'],torch=torch) as capture, torch.no_grad():
        model(input_ids=torch.tensor([[1,2,3,4]]))
        assert tuple(sorted(capture.activations)) == cfg.EXPECTED_PRESSURE_CAPTURE_NAMES
        assert all(t.shape == (1,4,2048) for t in capture.activations.values())


def test_checkpoint_policy_serializes_all_rng_only_at_recovery_steps(modules, tmp_path):
    cfg, boundary, _, training = modules
    class Model:
        def save_pretrained(self, target, safe_serialization):
            (target/'model.safetensors').write_bytes(b'test')
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    scaler = boundary.DynamicLossScaler(scale=4096,growth_interval=1000,hysteresis=2,minimum_scale=1)
    for step, include in [(0,False),(256,True),(712,True)]:
        target = training._save_checkpoint(model=Model(),optimizer=optimizer,scaler=scaler,step=step,
                    include_optimizer=include,root=tmp_path,schedule_hash=cfg.EXPECTED_SCHEDULE_SHA256,torch=torch)
        meta = json.loads((target/'checkpoint_metadata.json').read_text())
        assert meta['optimizer_saved'] == include
        assert (target/'training_state.pt').exists() == include
        if include:
            state = torch.load(target/'training_state.pt',weights_only=False)
            assert all(k in state for k in ['optimizer','loss_scaler','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_states'])


def test_pressure_scope_and_worker_duplication_fail_closed(modules):
    cfg, _, _, _ = modules
    config = cfg.load_config()
    config['conditions']['pressure_sites'] = ['h','q_post']
    with pytest.raises(ValueError,match='h-only'):
        cfg.validate_config(config)
    config = cfg.load_config()
    first = cfg.EXPECTED_CONDITION_IDS[0]
    config['runpod']['worker_assignments'][first] *= 2
    with pytest.raises(ValueError,match='exactly once'):
        cfg.validate_config(config)


class _Capture:
    def __init__(self):
        self.activations = {}

    def clear(self):
        self.activations.clear()

class _ToyModel(torch.nn.Module):
    def __init__(self, capture, *, omit_site=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        self.capture = capture
        self.omit_site = omit_site
        self.config = SimpleNamespace(num_hidden_layers=1)

    def forward(self, input_ids, labels):
        hidden = input_ids.float() * self.weight
        for site in ("h",):
            if site != self.omit_site:
                self.capture.activations[f"{site}.layer_0"] = hidden
        return SimpleNamespace(loss=(hidden - 0.25).square().mean())

def _pressure(run_config):
    return parse_pressure_config({
        "method": "orthogonal_l1",
        "sites": ["h"],
        "weight": 1.0,
        "step_budget": 0.2,
    })

def test_boundary_differentiates_exact_realized_h_only_objective(modules):
    run_config, boundary, _, _ = modules
    capture = _Capture()
    model = _ToyModel(capture)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    result = boundary.run_recipe_boundary(
        model=model,
        optimizer=optimizer,
        batches=[torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])],
        pressure=_pressure(run_config),
        capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0),
        gradient_clip_norm=0.05,
        torch=torch,
        device=torch.device("cpu"),
    )
    assert result["optimizer_step_skipped"] is False
    assert result["ol1_correction_applied"] is True
    assert result["pressure_capture_tensor_count"] == 1
    assert result["pressure_to_task_ratio_final"] <= 0.2 + 1e-9

    broken_capture = _Capture()
    broken = _ToyModel(broken_capture, omit_site="h")
    with pytest.raises(RuntimeError, match="pressure capture mismatch"):
        boundary.run_recipe_boundary(
            model=broken,
            optimizer=torch.optim.AdamW(broken.parameters(), lr=0.01),
            batches=[torch.tensor([[1.0, 2.0]])],
            pressure=_pressure(run_config),
            capture=broken_capture,
            loss_scaler=boundary.DynamicLossScaler(scale=8.0),
            gradient_clip_norm=1.0,
            torch=torch,
            device=torch.device("cpu"),
        )

def test_pressure_mean_is_not_divided_by_seven_site_count(modules):
    run_config, boundary, _, _ = modules
    capture = _Capture()
    model = _ToyModel(capture)
    result = boundary.run_recipe_boundary(
        model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        batches=[torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])],
        pressure=_pressure(run_config), capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0), gradient_clip_norm=1.0,
        torch=torch, device=torch.device("cpu"),
    )
    assert result["pressure_loss"] == pytest.approx(2.25)
