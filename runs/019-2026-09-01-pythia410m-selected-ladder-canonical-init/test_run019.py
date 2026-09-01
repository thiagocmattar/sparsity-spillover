from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import calibration
import calibration_comparison
import diagnostics
import initialization_artifact
import model_factory
import optimizer_boundary
import run_config
import teal_posthoc
import training
import verification


def test_condition_matrix_launch_wave_and_calibration_are_exact():
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert tuple(row["id"] for row in rows) == run_config.EXPECTED_CONDITION_IDS
    assert [row["order"] for row in rows] == list(range(1, 13))
    assert sum(row["topology_id"] == "A4-Z" for row in rows) == 5
    assert sum(row["topology_id"] == "A7-Z-POST" for row in rows) == 5
    assert tuple(config["runpod"]["launch_waves"]["all_conditions"]) == tuple(
        run_config.EXPECTED_CONDITION_IDS
    )
    assert tuple(config["calibration"]["condition_ids"]) == (
        "a0-gelu",
        "a4-ol1-kappa-0p5",
        "a7-ol1-kappa-0p5",
    )


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
def test_resolved_conditions_preserve_topology_and_gate_contract(
    condition_id, topology, sites, operators
):
    config = run_config.load_config()
    condition = next(row for row in run_config.condition_specs(config) if row["id"] == condition_id)
    resolved = run_config.resolved_condition_config(config, condition)
    assert resolved["model"]["topology_id"] == topology
    assert tuple(resolved["model"]["pressure_sites"]) == sites
    assert tuple(resolved["activation_pressure"]["sites"]) == sites
    gates = resolved["model"].get("site_gates")
    uniform = resolved["model"].get("site_gate")
    realized = (
        {spec["operator"] for spec in gates.values()}
        if gates
        else ({uniform["operator"]} if uniform else set())
    )
    assert realized == operators


def test_schedule_410m_ceilings_and_paired_endpoint_contract_are_pinned():
    config = run_config.load_config()
    train_metadata = json.loads(
        run_config.repo_path(config["data"]["training_metadata"]).read_text(encoding="utf-8")
    )
    _, schedule_sha256, _ = run_config.build_schedule(config, train_metadata, np=np)
    assert schedule_sha256 == run_config.EXPECTED_SCHEDULE_SHA256
    for topology, expected in run_config.EXPECTED_CEILINGS.items():
        ceiling = run_config.expected_ceiling(topology)
        assert (ceiling["reachable_product_count"], ceiling["model_product_count"]) == expected
        assert ceiling["R_model_max_fraction"] == expected[0] / expected[1]
    assert config["training"]["peak_learning_rate"] == 3e-4
    assert config["training"]["minimum_learning_rate"] == 3e-5
    assert config["training"]["activation_checkpointing"] is False
    assert config["validation"]["canonical_endpoint_loss"] == "logical_product_diagnostic_eager"
    assert config["diagnostics"]["activation_attention_implementation"] == "eager"
    assert config["teal_posthoc"]["diagnostic_sites"] == list(run_config.DIAGNOSTIC_SITES)


def test_artifact_and_science_contracts_fail_closed():
    config = run_config.load_config()
    for key, value in (
        ("model_sha256", "0" * 64),
        ("rng_bytes", config["initialization_artifact"]["rng_bytes"] + 1),
        ("metadata_sha256", "2" * 64),
        ("parameter_sha256", "1" * 64),
    ):
        changed = deepcopy(config)
        changed["initialization_artifact"][key] = value
        with pytest.raises(ValueError, match="canonical 410M initialization identity|not pinned"):
            run_config.validate_config(changed)
    changed = deepcopy(config)
    changed["model"]["released_weights_loaded"] = True
    with pytest.raises(ValueError, match="canonical random initialization"):
        run_config.validate_config(changed)
    changed = deepcopy(config)
    changed["training"]["activation_checkpointing"] = True
    with pytest.raises(ValueError, match="410M training decomposition"):
        run_config.validate_config(changed)


def test_hardware_and_guards_remain_unselected_until_calibration():
    config = run_config.load_config()
    runpod = config["runpod"]
    assert runpod["selected_gpu_type"] is None
    assert runpod["calibration_terminate_after_hours"] is None
    assert runpod["scientific_terminate_after_hours"] is None
    assert runpod["monitoring_interval_minutes"] == 10
    assert runpod["maximum_parallel_pods"] == 12
    changed = deepcopy(config)
    changed["runpod"]["scientific_terminate_after_hours"] = 6.5
    with pytest.raises(ValueError, match="pre-calibration"):
        run_config.validate_config(changed)


def test_generated_files_match_config_metadata_and_local_verification():
    config = run_config.load_config()
    artifact = config["initialization_artifact"]
    for prefix in ("model", "rng", "metadata"):
        path = run_config.RUN_DIR / artifact[f"{prefix}_path"]
        assert path.stat().st_size == artifact[f"{prefix}_bytes"]
        assert initialization_artifact.sha256_file(path) == artifact[f"{prefix}_sha256"]
    metadata = json.loads(
        (run_config.RUN_DIR / artifact["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["released_weights_loaded"] is False
    assert metadata["parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert metadata["roundtrip_parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert (metadata["tensor_count"], metadata["tensor_bytes"]) == (292, 1_621_336_064)
    verified = json.loads(
        (run_config.RUN_DIR / "prelaunch/initialization/verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert verified["status"] == "verified"
    assert verified["independent_strict_loads"] == 2
    assert verified["parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256


def test_generator_refuses_to_overwrite_canonical_artifacts():
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        initialization_artifact.generate_initialization_artifacts()


def test_strict_artifact_load_reproduces_parameters_and_post_init_rng():
    from transformers import AutoModelForCausalLM

    config = run_config.load_config()
    condition = next(
        row for row in run_config.condition_specs(config) if row["id"] == "a7-ol1-kappa-0p5"
    )
    resolved = run_config.resolved_condition_config(config, condition)
    model = model_factory.build_pinned_run019_model(
        resolved["model"],
        device=torch.device("cpu"),
        torch=torch,
        auto_model=AutoModelForCausalLM,
    )
    rng_path = run_config.RUN_DIR / config["initialization_artifact"]["rng_path"]
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(rng["python_rng_state"])
    np.random.set_state(rng["numpy_rng_state"])
    torch.set_rng_state(rng["torch_cpu_rng_state"])
    expected = (random.random(), float(np.random.random()), torch.rand(4))
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    recipe = initialization_artifact.load_pinned_initialization(model, torch=torch)
    observed = (random.random(), float(np.random.random()), torch.rand(4))
    assert run_config.parameter_sha256(model) == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert recipe["realization"] == "hash_pinned_generated_artifact"
    assert recipe["released_weights_loaded"] is False
    assert observed[:2] == expected[:2]
    assert torch.equal(observed[2], expected[2])
    assert sum(parameter.numel() for parameter in model.parameters()) == 405_334_016


def test_file_verification_rejects_byte_and_hash_drift(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"canonical")
    with pytest.raises(RuntimeError, match="byte count"):
        initialization_artifact._verify_file(path, expected_bytes=1, expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        initialization_artifact._verify_file(
            path, expected_bytes=len(b"canonical"), expected_sha256="0" * 64
        )


def test_worker_lifecycle_loads_artifact_before_hash_gate():
    assert training._BASE.apply_pythia_14m_initialization is initialization_artifact.load_pinned_initialization
    assert training._BASE.build_random_pythia is training._build_random_pythia
    assert training._BASE.parameter_sha256 is training._verified_initial_parameter_sha256


def test_training_rejects_parameter_mismatch_before_cuda_transfer(monkeypatch):
    model = torch.nn.Linear(2, 2)
    transfers = []
    hashes = []
    monkeypatch.setattr(training, "_transfer_initialized_model_to_cuda", transfers.append)

    def matching_hash(value):
        hashes.append(value)
        return run_config.EXPECTED_INITIAL_PARAMETER_SHA256

    monkeypatch.setattr(training, "_SOURCE_PARAMETER_SHA256", matching_hash)
    assert training._verified_initial_parameter_sha256(model) == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert transfers == [model]
    assert hashes == [model, model]
    monkeypatch.setattr(training, "_SOURCE_PARAMETER_SHA256", lambda model: "0" * 64)
    with pytest.raises(RuntimeError, match="before training"):
        training._verified_initial_parameter_sha256(model)
    assert transfers == [model]


def test_training_rejects_parameter_mismatch_after_cuda_transfer(monkeypatch):
    model = torch.nn.Linear(2, 2)
    transfers = []
    hashes = iter((run_config.EXPECTED_INITIAL_PARAMETER_SHA256, "0" * 64))
    monkeypatch.setattr(training, "_transfer_initialized_model_to_cuda", transfers.append)
    monkeypatch.setattr(training, "_SOURCE_PARAMETER_SHA256", lambda _model: next(hashes))
    with pytest.raises(RuntimeError, match="after the CPU-to-CUDA transfer"):
        training._verified_initial_parameter_sha256(model)
    assert transfers == [model]


def test_pressure_capture_names_scale_to_twenty_four_layers():
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[object()] * 24), config=SimpleNamespace())
    assert len(optimizer_boundary._expected_pressure_capture_names(model, run_config.A4_SITES)) == 96
    assert len(optimizer_boundary._expected_pressure_capture_names(model, run_config.A7_SITES)) == 168


def test_activation_diagnostic_forces_eager_and_restores_model(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    observed = {}

    def fake(**kwargs):
        observed["implementation"] = kwargs["model"].config._attn_implementation
        observed["context"] = diagnostics._BASE.recipe_attention_context
        return {"loss": 1.0}, {"site_definition": {}}

    monkeypatch.setattr(diagnostics._BASE, "activation_diagnostic_validation", fake)
    coverage, statistics = diagnostics.activation_diagnostic_validation(
        model=model, tokens=None, config={}, torch=None, np=None
    )
    assert coverage["loss"] == 1.0
    assert statistics["attention_implementation"] == "eager"
    assert observed == {
        "implementation": "eager",
        "context": diagnostics.eager_attention_context,
    }
    assert model.config._attn_implementation == "sdpa"


def test_teal_mapping_scales_to_twenty_four_layers():
    layers = []
    for _ in range(24):
        layers.append(
            SimpleNamespace(
                input_layernorm=object(),
                post_attention_layernorm=object(),
                mlp=SimpleNamespace(act=object()),
                attention=SimpleNamespace(dense=object()),
            )
        )
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))
    modules = teal_posthoc._module_map(model)
    assert len(modules) == 96
    assert set(name.split(".layer_", 1)[0] for name in modules) == set(run_config.A4_SITES)


def test_observed_r_model_is_not_bounded_by_topology_ceiling():
    verification._require_r_model_fraction(0.5, "synthetic")
    with pytest.raises(ValueError, match="outside"):
        verification._require_r_model_fraction(1.01, "synthetic")


def test_teal_threshold_ties_nondominance_and_complete_site_contract():
    values = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    thresholds = teal_posthoc.empirical_thresholds(values, (0.0, 0.5), np=np)
    assert thresholds["targets"][0]["calibration_zero_count"] == 1
    assert thresholds["targets"][1]["calibration_zero_count"] == 3
    assert set(run_config.DIAGNOSTIC_SITES) == {
        "a", "m", "h", "q_post", "k_post", "v", "z", "attention_output"
    }
    rows = [
        {"validation": {"loss": 2.0}, "logical_products": {"R_model": 0.1}},
        {"validation": {"loss": 2.1}, "logical_products": {"R_model": 0.2}},
        {"validation": {"loss": 2.2}, "logical_products": {"R_model": 0.05}},
    ]
    assert teal_posthoc.nondominated(rows) == [True, True, False]


def test_calibration_projection_covers_all_twelve_conditions():
    config = run_config.load_config()
    samples = []
    for condition_id, seconds in (
        ("a0-gelu", 10.0),
        ("a4-ol1-kappa-0p5", 20.0),
        ("a7-ol1-kappa-0p5", 30.0),
    ):
        sample = {
            "condition_id": condition_id,
            "setup_seconds": 5.0,
            "median_end_to_end_step_seconds": seconds,
            "min_end_to_end_step_seconds": seconds - 1.0,
            "max_end_to_end_step_seconds": seconds + 1.0,
        }
        if condition_id == "a0-gelu":
            sample["teal_probe"] = {
                "calibration_seconds": 11.0,
                "point": {"evaluation_seconds": 12.0},
            }
        if condition_id == "a7-ol1-kappa-0p5":
            sample.update(
                full_validation={"wall_seconds": 13.0},
                activation_diagnostic={"wall_seconds": 14.0},
                logical_diagnostic={"wall_seconds": 15.0},
                weight_diagnostic={"wall_seconds": 15.5},
                checkpoint={
                    "write_seconds": 16.0,
                    "inventory_seconds": 17.0,
                    "reload_seconds": 18.0,
                },
            )
        samples.append(sample)
    projection = calibration._project_science(
        config, samples, cache_verification_seconds=4.0, hourly_price_usd=1.0
    )
    assert set(projection["conditions"]) == set(run_config.EXPECTED_CONDITION_IDS)
    assert projection["conditions"]["a4-ol1-kappa-0"]["reference_condition_id"] == "a4-ol1-kappa-0p5"
    assert projection["conditions"]["a7-ol1-kappa-0"]["reference_condition_id"] == "a7-ol1-kappa-0p5"
    assert projection["projected_total_gpu_cost_usd"] == pytest.approx(
        projection["projected_gpu_hours"]
    )


def test_run_code_identity_is_direct_and_complete():
    identity = run_config.run_code_identity()
    paths = {row["path"] for row in identity["files"]}
    assert "initialization_artifact.py" in paths
    assert "calibration.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert not any("/017-" in path or "/018-" in path for path in paths)
    assert len(identity["content_sha256"]) == 64


def test_calibration_comparison_reports_cost_time_pareto_without_selecting(tmp_path):
    paths = []
    for index, (gpu, cost, hours) in enumerate(
        (("NVIDIA A40", 200.0, 40.0), ("NVIDIA A100-SXM4-80GB", 300.0, 20.0))
    ):
        artifact = {
            "status": "passed",
            "samples": [
                {
                    "initial_parameter_sha256": run_config.EXPECTED_INITIAL_PARAMETER_SHA256,
                    "peak_memory_reserved_bytes": 10 * 1024**3,
                }
            ],
            "cache": {"schedule_sha256": run_config.EXPECTED_SCHEDULE_SHA256},
            "run_code": {"content_sha256": "a" * 64},
            "price_snapshot": {
                "gpu_type_id": gpu,
                "cloud_type": "SECURE",
                "hourly_price_usd": 1.0 + index,
                "captured_at": "2026-09-01T00:00:00+00:00",
            },
            "device": {"name": gpu, "total_memory_bytes": 80 * 1024**3},
            "projection": {
                "projected_gpu_hours": 100.0,
                "projected_total_gpu_cost_usd": cost,
                "projected_twelve_way_makespan_seconds": hours * 3600.0,
                "twelve_gpu_hourly_burn_usd": 12.0 * (1.0 + index),
                "conditions": {},
            },
        }
        path = tmp_path / f"candidate-{index}.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "comparison.json"
    result = calibration_comparison.compare(paths, output)
    assert result["status"] == "ready_for_human_selection"
    assert result["selection"] is None
    assert result["cheapest_candidate"]["gpu_type_id"] == "NVIDIA A40"
    assert result["fastest_candidate"]["gpu_type_id"] == "NVIDIA A100-SXM4-80GB"
    assert all(row["nondominated_cost_makespan"] for row in result["candidates"])
