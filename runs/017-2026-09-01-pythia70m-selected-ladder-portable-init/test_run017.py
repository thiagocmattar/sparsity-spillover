from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import initialization
import model_factory
import optimizer_boundary
import run_config
import teal_posthoc
import training


def test_condition_matrix_and_parallel_waves_partition_exactly():
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert tuple(row["id"] for row in rows) == run_config.EXPECTED_CONDITION_IDS
    assert [row["order"] for row in rows] == list(range(1, 13))
    assert sum(row["topology_id"] == "A4-Z" for row in rows) == 5
    assert sum(row["topology_id"] == "A7-Z-POST" for row in rows) == 5
    waves = config["runpod"]["launch_waves"]
    assert set(waves["sentinel"]).isdisjoint(waves["remainder"])
    assert set(waves["sentinel"] + waves["remainder"]) == set(run_config.EXPECTED_CONDITION_IDS)


@pytest.mark.parametrize(
    ("condition_id", "topology", "sites", "operators"),
    [
        ("a0-gelu", "A0", (), set()),
        ("a1h-relu", "A1-H", (), {"relu"}),
        ("a4-ol1-kappa-0p1", "A4-Z", run_config.A4_SITES, {"one_sided_threshold"}),
        (
            "a7-ol1-kappa-0p1",
            "A7-Z-POST",
            run_config.A7_SITES,
            {"one_sided_threshold", "symmetric_threshold"},
        ),
    ],
)
def test_resolved_condition_is_fail_closed(condition_id, topology, sites, operators):
    config = run_config.load_config()
    condition = next(row for row in run_config.condition_specs(config) if row["id"] == condition_id)
    resolved = run_config.resolved_condition_config(config, condition)
    assert resolved["model"]["topology_id"] == topology
    assert tuple(resolved["model"]["pressure_sites"]) == sites
    assert tuple(resolved["activation_pressure"]["sites"]) == sites
    gates = resolved["model"].get("site_gates")
    uniform = resolved["model"].get("site_gate")
    realized = set(spec["operator"] for spec in gates.values()) if gates else ({uniform["operator"]} if uniform else set())
    assert realized == operators


def test_config_rejects_global_batch_decomposition_drift():
    config = run_config.load_config()
    changed = deepcopy(config)
    changed["training"]["gradient_accumulation_steps"] = 128
    with pytest.raises(ValueError, match="decomposition"):
        run_config.validate_config(changed)


def test_70m_analytic_ceilings_have_exact_integer_units():
    for topology, expected in run_config.EXPECTED_CEILINGS.items():
        row = run_config.expected_ceiling(topology)
        assert (row["reachable_product_count"], row["model_product_count"]) == expected
        assert row["R_model_max_fraction"] == expected[0] / expected[1]
        assert row["block_product_count"] == 51_545_899_008
        assert row["lm_head_product_count"] == 52_747_567_104


def test_pinned_schedule_and_portable_initialization_identities_are_sha256():
    assert len(run_config.EXPECTED_SCHEDULE_SHA256) == 64
    assert len(run_config.EXPECTED_INITIAL_PARAMETER_SHA256) == 64
    int(run_config.EXPECTED_SCHEDULE_SHA256, 16)
    int(run_config.EXPECTED_INITIAL_PARAMETER_SHA256, 16)


def test_pressure_capture_names_scale_with_model_layers_and_sites():
    model = SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=[object()] * 6),
        config=SimpleNamespace(),
    )
    names = optimizer_boundary._expected_pressure_capture_names(model, run_config.A7_SITES)
    assert len(names) == 42
    assert names == tuple(sorted(names))
    names = optimizer_boundary._expected_pressure_capture_names(model, run_config.A4_SITES)
    assert len(names) == 24


def test_deferred_microbatch_staging_preserves_global_batch():
    tokens = np.arange(64, dtype=np.int32)
    starts = np.asarray([[0, 4], [8, 12]])
    batches = training._microbatches_for_step(
        tokens,
        starts,
        block_size=4,
        device=torch.device("cpu"),
        torch=torch,
        np=np,
    )
    assert not isinstance(batches, list)
    realized = tuple(batches)
    assert len(realized) == 2
    assert all(tuple(batch.shape) == (2, 4) for batch in realized)
    assert sum(batch.numel() for batch in realized) == 16


def test_70m_initializer_accepts_only_70m_contract():
    class TinyModel(torch.nn.Module):
        def __init__(self, hidden_size=512):
            super().__init__()
            values = dict(initialization.EXPECTED_ARCHITECTURE)
            values["hidden_size"] = hidden_size
            self.config = SimpleNamespace(**values, hidden_dropout=0.0, attention_dropout=0.0)
            self.linear = torch.nn.Linear(3, 3)

    model = TinyModel()
    result = initialization.apply_pythia_70m_initialization(model, torch=torch)
    assert result["ordinary"] == "small_init"
    assert model.config.use_cache is False
    with pytest.raises(ValueError, match="Pythia-70M"):
        initialization.apply_pythia_70m_initialization(TinyModel(hidden_size=128), torch=torch)


def test_training_rejects_initialization_mismatch_before_boundary(monkeypatch):
    model = torch.nn.Linear(2, 2)
    transfers = []
    monkeypatch.setattr(training, "_transfer_initialized_model_to_cuda", transfers.append)
    monkeypatch.setattr(
        training,
        "_SOURCE_PARAMETER_SHA256",
        lambda model: run_config.EXPECTED_INITIAL_PARAMETER_SHA256,
    )
    assert (
        training._verified_initial_parameter_sha256(model)
        == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    )
    assert transfers == [model]

    monkeypatch.setattr(training, "_SOURCE_PARAMETER_SHA256", lambda model: "0" * 64)
    with pytest.raises(RuntimeError, match="before training"):
        training._verified_initial_parameter_sha256(model)
    assert transfers == [model]


def test_training_model_construction_forces_cpu_before_initialization(monkeypatch):
    captured = {}
    fake_model = SimpleNamespace(config=SimpleNamespace())

    def build(model_config, **kwargs):
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(training, "build_pinned_run017_model", build)
    realized = training._build_random_pythia(
        {"pressure_sites": ["a"]},
        device="cuda",
        torch=torch,
        auto_config=object(),
        auto_model=object(),
    )
    assert realized is fake_model
    assert captured["device"] == torch.device("cpu")
    assert fake_model.config.pressure_sites == ["a"]


def test_exact_70m_a7_graph_constructs_initializes_and_runs_forward():
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import topology_metadata

    config = run_config.load_config()
    condition = next(
        row for row in run_config.condition_specs(config) if row["id"] == "a7-ol1-kappa-0p5"
    )
    resolved = run_config.resolved_condition_config(config, condition)

    model = model_factory.build_pinned_run017_model(
        resolved["model"],
        device=torch.device("cpu"),
        torch=torch,
        auto_model=AutoModelForCausalLM,
    )
    torch.manual_seed(1234)
    initialization.apply_pythia_70m_initialization(model, torch=torch)
    assert run_config.parameter_sha256(model) == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    tokens = torch.randint(0, 50304, (1, 16))
    loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    assert sum(parameter.numel() for parameter in model.parameters()) == 70_426_624
    assert topology_metadata(model)["active_sites"] == list(run_config.A7_SITES)


def test_teal_mapping_uses_real_layer_count_not_14m_literal():
    def layer():
        return SimpleNamespace(
            input_layernorm=object(),
            post_attention_layernorm=object(),
            mlp=SimpleNamespace(act=object()),
            attention=SimpleNamespace(dense=object()),
        )

    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[layer()] * 9))
    modules = teal_posthoc._module_map(model)
    assert len(modules) == 9 * 4
    assert "z.layer_8" in modules


def test_teal_empirical_thresholds_preserve_tie_semantics():
    values = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    result = teal_posthoc.empirical_thresholds(values, (0.0, 0.5), np=np)
    assert result["targets"][0]["threshold"] == 0.0
    assert result["targets"][0]["calibration_zero_count"] == 1
    assert result["targets"][1]["threshold"] == 1.0
    assert result["targets"][1]["calibration_zero_count"] == 3


def test_nondominated_prefers_lower_loss_and_higher_rmodel():
    rows = [
        {"validation": {"loss": 2.0}, "logical_products": {"R_model": 0.1}},
        {"validation": {"loss": 2.1}, "logical_products": {"R_model": 0.2}},
        {"validation": {"loss": 2.2}, "logical_products": {"R_model": 0.05}},
    ]
    assert teal_posthoc.nondominated(rows) == [True, True, False]
