from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import initialization_artifact
import model_factory
import optimizer_boundary
import run_config
import teal_posthoc
import training


def test_condition_matrix_and_waves_are_the_approved_twelve():
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert tuple(row["id"] for row in rows) == run_config.EXPECTED_CONDITION_IDS
    assert [row["order"] for row in rows] == list(range(1, 13))
    assert sum(row["topology_id"] == "A4-Z" for row in rows) == 5
    assert sum(row["topology_id"] == "A7-Z-POST" for row in rows) == 5
    waves = config["runpod"]["launch_waves"]
    assert tuple(waves["sentinel"]) == run_config.EXPECTED_SENTINEL
    assert tuple(waves["remainder"]) == run_config.EXPECTED_REMAINDER
    assert set(waves["sentinel"]).isdisjoint(waves["remainder"])


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


def test_science_schedule_ceilings_and_diagnostics_remain_pinned():
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
    assert config["validation"] == {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692224,
        "excluded_tail_tokens": 1444,
        "batch_size": 4,
        "at_step_one": True,
        "at_final_step": True,
    }
    assert config["teal_posthoc"]["condition_ids"] == ["a0-gelu", "a1h-relu"]


def test_artifact_contract_is_fail_closed():
    config = run_config.load_config()
    for key, value in (
        ("model_sha256", "0" * 64),
        ("rng_bytes", config["initialization_artifact"]["rng_bytes"] + 1),
    ):
        changed = deepcopy(config)
        changed["initialization_artifact"][key] = value
        with pytest.raises(ValueError, match="canonical initialization artifact identity"):
            run_config.validate_config(changed)

    changed = deepcopy(config)
    changed["model"]["released_weights_loaded"] = True
    with pytest.raises(ValueError):
        run_config.validate_config(changed)


def test_science_guard_is_unset_until_timing_preflight():
    config = run_config.load_config()
    assert config["runpod"]["preflight_terminate_after_hours"] == 1.5
    assert config["runpod"]["scientific_terminate_after_hours"] is None
    changed = deepcopy(config)
    changed["runpod"]["scientific_terminate_after_hours"] = 6.5
    with pytest.raises(ValueError, match="timing-only preflight"):
        run_config.validate_config(changed)


def test_generated_files_match_config_and_tracked_metadata():
    config = run_config.load_config()
    artifact = config["initialization_artifact"]
    for prefix in ("model", "rng"):
        path = run_config.RUN_DIR / artifact[f"{prefix}_path"]
        assert path.stat().st_size == artifact[f"{prefix}_bytes"]
        assert initialization_artifact.sha256_file(path) == artifact[f"{prefix}_sha256"]
    metadata = json.loads(
        (run_config.RUN_DIR / artifact["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["released_weights_loaded"] is False
    assert metadata["parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert metadata["roundtrip_parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert (metadata["tensor_count"], metadata["tensor_bytes"]) == (76, 281_706_496)


def test_generator_refuses_to_overwrite_canonical_artifacts():
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        initialization_artifact.generate_initialization_artifacts()


def test_strict_artifact_load_reproduces_parameters_and_post_init_rng():
    from transformers import AutoModelForCausalLM

    config = run_config.load_config()
    condition = next(row for row in run_config.condition_specs(config) if row["id"] == "a7-ol1-kappa-0")
    resolved = run_config.resolved_condition_config(config, condition)
    model = model_factory.build_pinned_run018_model(
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
    assert sum(parameter.numel() for parameter in model.parameters()) == 70_426_624


def test_file_verification_rejects_byte_and_hash_drift(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"canonical")
    with pytest.raises(RuntimeError, match="byte count"):
        initialization_artifact._verify_file(path, expected_bytes=1, expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        initialization_artifact._verify_file(
            path, expected_bytes=len(b"canonical"), expected_sha256="0" * 64
        )


def test_worker_lifecycle_uses_artifact_loader_before_existing_hash_gate():
    assert (
        training._FROZEN._BASE.apply_pythia_14m_initialization
        is initialization_artifact.load_pinned_initialization
    )
    assert training._FROZEN._BASE.build_random_pythia is training._FROZEN._build_random_pythia
    assert (
        training._FROZEN._BASE.parameter_sha256
        is training._FROZEN._verified_initial_parameter_sha256
    )


def test_training_rejects_parameter_mismatch_before_transfer(monkeypatch):
    model = torch.nn.Linear(2, 2)
    transfers = []
    frozen = training._FROZEN
    monkeypatch.setattr(frozen, "_transfer_initialized_model_to_cuda", transfers.append)
    monkeypatch.setattr(
        frozen, "_SOURCE_PARAMETER_SHA256", lambda model: run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    )
    assert frozen._verified_initial_parameter_sha256(model) == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert transfers == [model]
    monkeypatch.setattr(frozen, "_SOURCE_PARAMETER_SHA256", lambda model: "0" * 64)
    with pytest.raises(RuntimeError, match="before training"):
        frozen._verified_initial_parameter_sha256(model)
    assert transfers == [model]


def test_pressure_capture_names_scale_to_six_layers():
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[object()] * 6), config=SimpleNamespace())
    assert len(optimizer_boundary._expected_pressure_capture_names(model, run_config.A4_SITES)) == 24
    assert len(optimizer_boundary._expected_pressure_capture_names(model, run_config.A7_SITES)) == 42


def test_teal_threshold_ties_and_nondominance_are_inherited():
    values = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    thresholds = teal_posthoc.empirical_thresholds(values, (0.0, 0.5), np=np)
    assert thresholds["targets"][0]["calibration_zero_count"] == 1
    assert thresholds["targets"][1]["calibration_zero_count"] == 3
    rows = [
        {"validation": {"loss": 2.0}, "logical_products": {"R_model": 0.1}},
        {"validation": {"loss": 2.1}, "logical_products": {"R_model": 0.2}},
        {"validation": {"loss": 2.2}, "logical_products": {"R_model": 0.05}},
    ]
    assert teal_posthoc.nondominated(rows) == [True, True, False]


def test_run_code_identity_covers_every_frozen_layer():
    identity = run_config.run_code_identity()
    paths = {row["path"] for row in identity["files"]}
    assert "initialization_artifact.py" in paths
    assert "../017-2026-09-01-pythia70m-selected-ladder-portable-init/training.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert len(identity["content_sha256"]) == 64
