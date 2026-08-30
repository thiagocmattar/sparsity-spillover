import importlib.util
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparsity_research.pressure import parse_pressure_config


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "004-2026-08-29-pythia14m-full-pass-l1n"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, RUN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _figure_module():
    name = "run004_spillover_figures"
    path = RUN_DIR / "07_plot_spillover_figures.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    names = ("run_config", "initialization", "optimizer_boundary", "smoke")
    previous = {name: sys.modules.get(name) for name in names}
    sys.path.insert(0, str(RUN_DIR))
    try:
        run_config = _module("run_config")
        initialization = _module("initialization")
        boundary = _module("optimizer_boundary")
        smoke = _module("smoke")
        yield run_config, initialization, boundary, smoke
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_six_condition_recipe_contract_is_locked(modules):
    run_config, _, _, _ = modules
    config = run_config.load_config()
    assert config["training"] == {
        "max_steps": 712,
        "peak_learning_rate": 1e-3,
        "minimum_learning_rate": 1e-4,
        "global_batch_size": 1024,
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 32,
        "optimizer": "adamw",
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "warmup_fraction": 0.01,
        "precision": "float16_dynamic",
        "parameter_dtype": "float32",
        "initial_loss_scale_power": 12,
        "loss_scale_window": 1000,
        "loss_scale_hysteresis": 2,
        "minimum_loss_scale": 1.0,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_checkpointing": False,
        "device": "cuda",
        "log_every_steps": 1,
    }
    rows = run_config.condition_specs(config)
    assert [row["id"] for row in rows] == list(run_config.EXPECTED_CONDITION_IDS)
    assert [row["pressure_weight"] for row in rows] == [0.0, 0.0, 0.05, 0.1, 0.5, 1.0]
    assert run_config.worker_conditions(config, "controls") == rows[:2]
    assert config["recipe"]["exact_framework_reproduction"] is False
    assert config["recipe"]["weight_decay_exclusions"] == ["bias", "layer_norm"]
    assert config["recipe"]["lr_schedule_semantics"] == "gpt_neox_v1_pre_step"
    assert config["runtime"] == {
        "python": "3.12",
        "torch": "2.11.0",
        "transformers": "5.12.1",
        "cuda_runtime": "12.8",
    }
    assert config["runpod"]["image"].startswith("runpod/pytorch@sha256:")
    assert config["runpod"]["terminate_after_hours"] == 7


def test_gate_and_pressure_are_independent_fields(modules):
    run_config, _, _, _ = modules
    config = run_config.load_config()
    gelu, relu, pressure = run_config.condition_specs(config)[:3]
    gelu_resolved = run_config.resolved_condition_config(config, gelu)
    relu_resolved = run_config.resolved_condition_config(config, relu)
    pressure_resolved = run_config.resolved_condition_config(config, pressure)
    assert gelu_resolved["model"]["topology_id"] == "A0"
    assert gelu_resolved["model"]["site_gate"] is None
    assert relu_resolved["model"]["topology_id"] == "A1-H"
    assert relu_resolved["model"]["site_gate"] == {"operator": "relu"}
    assert relu_resolved["activation_pressure"]["method"] == "none"
    assert pressure_resolved["activation_pressure"] == {
        "method": "l1_naive",
        "sites": ["h"],
        "weight": 0.05,
        "step_budget": None,
        "eps": 1e-12,
    }


def test_full_pass_schedule_and_validation_arithmetic(modules):
    run_config, _, _, _ = modules
    config = run_config.load_config()
    metadata = {
        "tokens": run_config.TRAIN_CACHE_TOKENS,
        "documents": run_config.TRAIN_CACHE_DOCUMENTS,
        "tokens_sha256": run_config.TRAIN_CACHE_SHA256,
        "block_size": 2048,
        "split": "train",
    }
    _, _, schedule = run_config.build_schedule(config, metadata, np=np)
    assert schedule["complete_blocks"] == 728_374
    assert schedule["scheduled_blocks"] == 729_088
    assert schedule["wrapped_blocks"] == 714
    assert 712 * 1024 * 2048 == 1_493_172_224
    run_config.require_validation_coverage(
        {
            "sequences": 338,
            "input_tokens": 692_224,
            "excluded_tail_tokens": 1_444,
            "complete_block_coverage": True,
        },
        config,
    )


def test_run_code_identity_includes_dirty_shared_scientific_modules(modules):
    run_config, _, _, _ = modules
    identity = run_config.run_code_identity()
    names = [row["path"] for row in identity["files"]]
    assert "optimizer_boundary.py" in names
    assert "../../src/sparsity_research/pressure.py" in names
    assert "../../src/sparsity_research/pythia.py" in names
    assert len(identity["content_sha256"]) == 64


def test_pythia_initialization_uses_small_and_wang_roles(modules):
    _, initialization, _, _ = modules
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        max_position_embeddings=2048,
        rotary_pct=0.25,
        tie_word_embeddings=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_parallel_residual=True,
        hidden_act="gelu",
    )
    torch.manual_seed(1234)
    model = GPTNeoXForCausalLM(config)
    torch.manual_seed(1234)
    metadata = initialization.apply_pythia_14m_initialization(model, torch=torch)
    initialization.verify_recipe_model(model)
    small = math.sqrt(2.0 / (5.0 * 128))
    wang = 2.0 / 6.0 / math.sqrt(128)
    assert metadata["small_init_std"] == pytest.approx(small)
    assert metadata["wang_init_std"] == pytest.approx(wang)
    ordinary = model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight.detach().std().item()
    residual = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.detach().std().item()
    assert ordinary == pytest.approx(small, rel=0.03)
    assert residual == pytest.approx(wang, rel=0.03)
    assert torch.count_nonzero(model.gpt_neox.layers[0].mlp.dense_4h_to_h.bias) == 0


class _Capture:
    def __init__(self):
        self.activations = {}

    def clear(self):
        self.activations.clear()


class _ToyModel(torch.nn.Module):
    def __init__(self, capture):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.capture = capture

    def forward(self, input_ids, labels):
        activation = self.weight * input_ids.float()
        self.capture.activations["h.layer_0"] = activation
        return SimpleNamespace(loss=activation.square().mean())


def test_fp16_recipe_boundary_matches_naive_l1_gradient_on_cpu(modules):
    _, _, boundary, _ = modules
    capture = _Capture()
    model = _ToyModel(capture)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pressure = parse_pressure_config(
        {"method": "l1_naive", "sites": ["h"], "weight": 0.5, "eps": 1e-12}
    )
    result = boundary.run_recipe_boundary(
        model=model,
        optimizer=optimizer,
        batches=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])],
        pressure=pressure,
        capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0),
        gradient_clip_norm=100.0,
        torch=torch,
        device=torch.device("cpu"),
    )
    # Equal microbatch weighting: task grad=(5*2 + 18*2)/2=23; L1 grad=(1.5+3)/2=2.25.
    assert model.weight.item() == pytest.approx(2.0 - 0.1 * (23.0 + 0.5 * 2.25))
    assert result["task_gradient_norm"] == pytest.approx(23.0)
    assert result["pressure_gradient_norm"] == pytest.approx(2.25)
    assert result["adamw_gradient_was_clipped"] is False
    assert result["optimizer_step_skipped"] is False


def test_dynamic_loss_scaler_honors_growth_window_and_hysteresis(modules):
    _, _, boundary, _ = modules
    scaler = boundary.DynamicLossScaler(scale=8.0, growth_interval=2, hysteresis=2)
    assert scaler.update(finite=True)["loss_scale_action"] == "hold"
    assert scaler.update(finite=True)["loss_scale"] == 16.0
    assert scaler.update(finite=False)["loss_scale_action"] == "overflow_hold"
    backed_off = scaler.update(finite=False)
    assert backed_off["loss_scale_action"] == "backoff"
    assert backed_off["loss_scale"] == 8.0


def test_recipe_optimizer_excludes_biases_and_layernorm_from_decay(modules):
    run_config, _, boundary, _ = modules

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.norm = torch.nn.LayerNorm(3)

    model = Tiny()
    optimizer, metadata = boundary.build_recipe_adamw(
        model, run_config.load_config()["training"], torch=torch
    )
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]
    assert metadata["decay_parameter_names"] == ["linear.weight"]
    assert set(metadata["no_decay_parameter_names"]) == {
        "linear.bias", "norm.weight", "norm.bias"
    }


def test_recipe_learning_rate_matches_gpt_neox_v1_pre_step_semantics(modules):
    _, _, boundary, _ = modules
    kwargs = dict(peak=1e-3, max_steps=712, warmup_fraction=0.01, minimum=1e-4)
    assert boundary.recipe_learning_rate(1, **kwargs) == 0.0
    assert boundary.recipe_learning_rate(2, **kwargs) == pytest.approx(1e-3 / 7.12)
    assert boundary.recipe_learning_rate(8, **kwargs) == pytest.approx(7e-3 / 7.12)
    assert boundary.recipe_learning_rate(712, **kwargs) == 1e-4


def test_smoke_projection_never_claims_an_exact_fit(modules):
    _, _, _, smoke = modules
    samples = [
        {"condition_id": name, "status": "completed", "batch_size": batch, "peak_memory_allocated_bytes": memory}
        for name in ("relu-control", "relu-l1n-1")
        for batch, memory in ((2, 2_000_000_000), (4, 3_000_000_000))
    ]
    projection = smoke._project_target(
        samples,
        {"micro_batch_size": 32, "memory_bytes": 24 * 1024**3},
    )
    for row in projection["conditions"].values():
        assert row["status"] == "projected"
        assert "not a substitute" in row["caveat"]


def test_exact_target_smoke_requires_both_complete_paths_and_headroom(modules):
    _, _, _, smoke = modules
    target = {
        "micro_batch_size": 32,
        "sequence_length": 2048,
        "gradient_accumulation_steps": 32,
        "memory_bytes": 24 * 1024**3,
    }
    samples = [
        {
            "condition_id": name,
            "status": "completed",
            "batch_size": 32,
            "sequence_length": 2048,
            "gradient_accumulation_steps": 32,
            "peak_memory_reserved_bytes": 20 * 1024**3,
            "boundary_seconds": [10.0, 9.5],
            "boundary_health": [
                {"optimizer_step_skipped": False, "gradient_overflow": False},
                {"optimizer_step_skipped": False, "gradient_overflow": False},
            ],
        }
        for name in ("relu-control", "relu-l1n-1")
    ]
    exact = smoke._exact_target(samples, target, 2048, 32)
    assert exact["status"] == "sampled"
    assert exact["all_conditions_fit"] is True
    samples[1]["peak_memory_reserved_bytes"] = 23 * 1024**3
    assert smoke._exact_target(samples, target, 2048, 32)["all_conditions_fit"] is False


def test_launch_plan_cost_envelope_and_staging_are_explicit():
    plan = json.loads((RUN_DIR / "prelaunch" / "launch-plan.json").read_text(encoding="utf-8"))
    assert isinstance(plan["status"], str) and plan["status"]
    capacity_attempt = plan["launch_attempts"][0]
    assert capacity_attempt["result"] == "capacity_unavailable"
    assert capacity_attempt["created_pods_after_attempt"] == 0
    assert capacity_attempt["created_network_volumes_after_attempt"] == 0
    assert capacity_attempt["billable_cost_usd"] == 0.0
    assert plan["staged_execution"]["stage_one"]["pod_count"] == 2
    assert plan["staged_execution"]["stage_two"]["pod_count_total"] == 5
    assert plan["staged_execution"]["stage_two"]["cloud_type"] == "SECURE"
    assert plan["staged_execution"]["stage_two"]["network_volume_id"] == "9luykg5yc3"
    assert plan["estimate"]["maximum_compute_usd"] == pytest.approx(5 * 7 * 0.74 + 7 * 0.34)
    assert plan["estimate"]["maximum_total_usd"] == 28.55
    assert plan["persistent_network_volume"]["price_usd_per_month"] == 7.0
    assert plan["teardown"]["continuing_resources_after_success"] == [
        {
            "type": "network_volume",
            "id": "9luykg5yc3",
            "name": "sparsity-spillover-shared",
            "monthly_cost_usd": 7.0,
            "reason": "user-requested reusable cache and artifact storage",
        }
    ]


def _fake_activation_statistics(hits_per_layer):
    sites = ("h", "q_post", "k_post", "v", "m", "attention_output")
    return {
        "pooled_by_site": [
            {
                "name": site,
                "threshold_hits": {"0.001": 6 * hits_per_layer},
                "total": 60,
                "nonfinite": 0,
                # Deliberately wrong: the plotting reducer must ignore percentages.
                "threshold_fractions": {"0.001": 0.999},
            }
            for site in sites
        ],
        "rows": [
            {
                "name": f"{site}.layer_{layer}",
                "threshold_hits": {"0.001": hits_per_layer},
                "total": 10,
                "nonfinite": 0,
            }
            for site in sites
            for layer in range(6)
        ],
    }


def test_spillover_figure_reduction_pools_integer_counts_before_dividing():
    figures = _figure_module()
    statistics = _fake_activation_statistics(1)
    pooled = figures._pooled_counts(statistics, "h")
    assert pooled == {
        "hits": 6,
        "total": 60,
        "fraction": pytest.approx(0.1),
        "percent": pytest.approx(10.0),
    }

    statistics["rows"][0]["threshold_hits"]["0.001"] = 2
    with pytest.raises(ValueError, match="Layer and pooled counts disagree"):
        figures._pooled_counts(statistics, "h")


def test_spillover_figure_reduction_preserves_control_and_relu_dose_topology():
    figures = _figure_module()
    specifications = [
        ("gelu-control", "gelu", True, 0.0),
        ("relu-control", "relu", True, 0.0),
        ("relu-l1n-0p05", "relu", False, 0.05),
        ("relu-l1n-0p1", "relu", False, 0.1),
        ("relu-l1n-0p5", "relu", False, 0.5),
        ("relu-l1n-1", "relu", False, 1.0),
    ]
    conditions = []
    statistics = {}
    for order, (condition_id, activation, is_control, pressure_weight) in enumerate(
        specifications
    ):
        attempt_id = f"attempt-{order}"
        conditions.append(
            {
                "attempt_id": attempt_id,
                "final_validation_loss": 5.0 + order / 100.0,
                "condition": {
                    "order": order,
                    "id": condition_id,
                    "activation": activation,
                    "is_control": is_control,
                    "pressure_weight": pressure_weight,
                },
            }
        )
        statistics[attempt_id] = _fake_activation_statistics(order + 1)

    grouped = figures.reduce_points(
        {"status": "verified", "condition_count": 6, "conditions": conditions},
        statistics.__getitem__,
    )
    assert [point["pressure_weight"] for point in grouped["gelu"]] == [0.0]
    assert [point["pressure_weight"] for point in grouped["relu"]] == [
        0.0,
        0.05,
        0.1,
        0.5,
        1.0,
    ]
    assert grouped["gelu"][0]["near_zero"]["h"]["percent"] == pytest.approx(
        10.0
    )


def test_closed_run_reconciles_science_runpod_and_interpretation_state():
    closeout = json.loads(
        (RUN_DIR / "artifacts" / "closeout.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (RUN_DIR / "artifacts" / "verification.json").read_text(encoding="utf-8")
    )
    assert closeout["status"] == "closed"
    assert closeout["evidence_label"] == "valid"
    assert verification["status"] == "verified"
    assert closeout["scientific_execution"]["condition_count"] == 6
    assert closeout["scientific_execution"]["completed_optimizer_steps"] == 4272
    assert closeout["scientific_execution"]["training_input_tokens"] == 8_959_033_344
    assert closeout["runpod_closeout"]["active_pods"] == []
    assert closeout["runpod_closeout"]["active_gpu_cost_usd_per_hour"] == 0.0
    assert closeout["runpod_closeout"]["retained_network_volumes"] == [
        {
            "id": "9luykg5yc3",
            "name": "sparsity-spillover-shared",
            "data_center_id": "EUR-IS-1",
            "size_gb": 100,
            "continuing_cost_usd_per_month": 7.0,
            "retention": "intentional; user-requested reusable cache and artifact storage",
        }
    ]
    assert closeout["runpod_closeout"]["posted_pod_billing"][
        "total_usd"
    ] == pytest.approx(21.52134521584958)
    assert closeout["scientific_interpretation"] == {
        "observations_complete": True,
        "consolidated_finding": False,
        "manuscript_updated": False,
        "scope": "one seed, Pythia-14M, one full MiniPile pass",
    }
    assert len(closeout["remaining_post_hoc_todos"]) == 1
