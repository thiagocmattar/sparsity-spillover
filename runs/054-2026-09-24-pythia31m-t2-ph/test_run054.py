"""Scientific identity, gate placement, data coverage and optimizer behavior."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import transformers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_config import (load_config, condition_specs, resolved_condition_config,
    build_schedule, parameter_sha256, validate_config, require_validation_coverage)
from model_factory import build_pinned_run054_model
from initialization_artifact import load_pinned_initialization
from sparsity_research.capture import ActivationCapture
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.ceilings import architecture_ceiling
from optimizer_boundary import build_recipe_adamw, DynamicLossScaler, run_recipe_boundary

torch.set_num_threads(2)


def build(row):
    config = resolved_condition_config(load_config(), row)
    model = build_pinned_run054_model(config["model"], device=torch.device("cpu"),
        torch=torch, auto_model=transformers.AutoModelForCausalLM)
    load_pinned_initialization(model, torch=torch)
    return config, model


def test_recipe_matches_executed_scales():
    import yaml
    cfg = load_config()
    assert [r["gate_threshold"] for r in condition_specs(cfg)] == [None, 0., .01, .05, .1, .5]
    for folder in ("041-2026-09-20-pythia14m-hz-h-only-ol1", "043-2026-09-20-pythia70m-hz-h-only-ol1"):
        old = yaml.safe_load((HERE.parent/folder/"config.yaml").read_text())
        for section in ("seeds", "diagnostics", "checkpoints"):
            assert all(cfg[section][key] == value for key, value in old[section].items())
        for key, value in old["training"].items():
            if key not in ("micro_batch_size", "gradient_accumulation_steps"):
                assert cfg["training"][key] == value, key
        for key in ("dataset", "revision", "tokenizer", "tokenizer_revision", "append_eos", "sequence_length"):
            assert cfg["data"][key] == old["data"][key]


def test_data_order_and_coverage():
    cfg = load_config()
    starts, digest, metadata = build_schedule(cfg, {"tokens":1491711416}, np=np)
    assert starts.shape == (712, cfg["training"]["gradient_accumulation_steps"], cfg["training"]["micro_batch_size"])
    assert metadata["scheduled_blocks"] == 729088 and metadata["wrapped_blocks"] == 714
    other = deepcopy(cfg)
    other["training"].update(micro_batch_size=32, gradient_accumulation_steps=32)
    legacy, old_hash, _ = build_schedule(other, {"tokens":1491711416}, np=np)
    assert old_hash == "f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa"
    assert np.array_equal(starts.reshape(-1), legacy.reshape(-1))
    assert len(digest) == 64
    with pytest.raises(RuntimeError):
        require_validation_coverage(dict(sequences=337, input_tokens=690176,
            excluded_tail_tokens=1444, complete_block_coverage=True), cfg)


@pytest.mark.parametrize("index", range(6))
def test_initializer_and_exact_gate_ports(index):
    cfg, model = build(condition_specs(load_config())[index])
    meta = json.loads((HERE/"prelaunch/initialization/metadata.json").read_text())
    assert sum(p.numel() for p in model.parameters()) == 30494720
    assert parameter_sha256(model) == meta["parameter_sha256"]
    sites = topology_metadata(model)["active_sites"]
    assert list(sites) == ([] if index == 0 else ["h", "z"])
    if index:
        k = cfg["condition"]["gate_threshold"]
        for layer in model.gpt_neox.layers:
            for gate in (layer.mlp.act, layer.attention.z_gate):
                x = torch.tensor([-1., k-.0001, k, k+.1], requires_grad=True)
                y = gate(x)
                assert torch.equal(y, torch.tensor([0., 0., k, k+.1]))
                y.sum().backward()
                assert torch.equal(x.grad, torch.tensor([0., 0., 1., 1.]))
        model.eval()
        with ActivationCapture(model, ["h", "z"], torch=torch) as capture:
            model(input_ids=torch.arange(16).reshape(1,-1))
            assert set(capture.activations) == {f"{s}.layer_{i}" for s in ("h", "z") for i in range(6)}
            assert all(torch.all((v == 0) | (v >= k)) for v in capture.activations.values())


@pytest.mark.parametrize("index", [0, 3])
def test_optimizer_boundary_and_checkpoint_reload(index, tmp_path):
    cfg, model = build(condition_specs(load_config())[index])
    optimizer, _ = build_recipe_adamw(model, cfg["training"], torch=torch)
    pressure = parse_pressure_config(cfg["activation_pressure"])
    from contextlib import nullcontext
    context = ActivationCapture(model, ["h"], torch=torch) if pressure.enabled else nullcontext(None)
    with context as capture:
        result = run_recipe_boundary(model=model, optimizer=optimizer,
            batches=[torch.arange(16).reshape(1,-1)], pressure=pressure, capture=capture,
            loss_scaler=DynamicLossScaler(), gradient_clip_norm=1., torch=torch, device=torch.device("cpu"))
    assert not result["optimizer_step_skipped"]
    if index:
        assert result["pressure_capture_tensor_count"] == 6
        assert result["pressure_sites"] == ["h"] and result["ol1_correction_applied"]
        assert result["pressure_to_task_ratio_final"] <= 1.+1e-9
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=torch.arange(16).reshape(1,-1)).logits
    model.save_pretrained(tmp_path)
    restored = load_checkpoint_pythia(transformers.AutoModelForCausalLM, tmp_path, torch=torch).eval()
    assert topology_metadata(model) == topology_metadata(restored)
    with torch.no_grad():
        assert torch.equal(expected, restored(input_ids=torch.arange(16).reshape(1,-1)).logits)


def test_ceiling_and_validator_do_not_freeze_hardware_choices():
    value = architecture_ceiling("HZ", layers=6, hidden_size=256, ffn_size=1024,
        sequence_length=2048, vocabulary_size=50304)
    assert value["reachable_product_count"] == 6*2048*(256*1024+256*256)
    cfg = load_config()
    cfg["training"].update(micro_batch_size=2, gradient_accumulation_steps=512,
                           peak_learning_rate=.002)
    validate_config(cfg)
    cfg["conditions"]["gate_thresholds"] = [float("nan")]
    with pytest.raises(ValueError):
        validate_config(cfg)
