from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels"


def load_module(alias: str, path: Path):
    specification = importlib.util.spec_from_file_location(alias, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[alias] = module
    specification.loader.exec_module(module)
    return module


sys.path.insert(0, str(RUN_DIR))
try:
    RUN_CONFIG = load_module("run024_run_config", RUN_DIR / "run_config.py")
    CORE = load_module("run024_benchmark_core", RUN_DIR / "benchmark_core.py")
    SPARSE = load_module("run024_pythia_sparse", RUN_DIR / "pythia_sparse.py")
finally:
    sys.path.remove(str(RUN_DIR))


def load_script(name: str):
    path = RUN_DIR / name
    specification = importlib.util.spec_from_file_location(f"run024_{path.stem}", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    previous = {
        name: sys.modules.get(name)
        for name in ("run_config", "benchmark_core", "pythia_sparse")
    }
    sys.modules["run_config"] = RUN_CONFIG
    sys.modules["benchmark_core"] = CORE
    sys.modules["pythia_sparse"] = SPARSE
    sys.path.insert(0, str(RUN_DIR))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(RUN_DIR))
        for module_name, value in previous.items():
            if value is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = value
    return module


def test_config_is_exact_six_condition_410m_scope() -> None:
    config = RUN_CONFIG.load_config()
    assert tuple(condition["id"] for condition in config["conditions"]) == RUN_CONFIG.EXPECTED_SENTINELS
    assert tuple(config["condition_partition"]["sentinel"]) == RUN_CONFIG.EXPECTED_SENTINELS
    assert config["model"]["size"] == "410m"
    assert config["model"]["parameter_count"] == 405_334_016
    assert config["model"]["architecture"] == {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "layers": 24,
        "attention_heads": 16,
        "head_size": 64,
        "sequence_length": 2048,
        "vocab_size": 50304,
        "attention_bias": True,
        "mlp_bias": True,
    }
    assert config["measurement"]["full_model_batches"] == [1]
    assert config["measurement"]["optional_full_model_batch"] == 32
    assert config["validation"]["runtime_equivalence_batch_size"] == 1


def test_linear_shape_specialization_covers_pythia410m() -> None:
    assert CORE.LINEAR_OPERATION_SPECS == {
        "qkv_projection": {"site": "a", "K": 1024, "N": 3072},
        "mlp_w1": {"site": "m", "K": 1024, "N": 4096},
        "mlp_w2": {"site": "h", "K": 4096, "N": 1024},
        "attention_output_projection": {"site": "z", "K": 1024, "N": 1024},
    }
    remote_preflight = (RUN_DIR / "02_remote_preflight.py").read_text(encoding="utf-8")
    for shape in (
        '("qkv_projection", 256, 1024, 3072)',
        '("mlp_w1", 256, 1024, 4096)',
        '("mlp_w2", 256, 4096, 1024)',
        '("attention_output_projection", 256, 1024, 1024)',
    ):
        assert shape in remote_preflight


def test_transfer_allowlist_contains_six_model_only_checkpoints() -> None:
    config = RUN_CONFIG.load_config()
    prepare = load_script("00_prepare_inputs.py")
    paths = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in prepare.transfer_files(config, "sentinel")
    ]
    assert sum(path.endswith("model.safetensors") for path in paths) == 6
    assert sum(path.endswith("logical_products.json") for path in paths) == 6
    assert any(path.endswith("sakana-pythia70.patch") for path in paths)
    assert any(path.endswith("validation/tokens.int32.bin") for path in paths)
    assert not any("training_state.pt" in path for path in paths)
    assert not any("optimizer" in path.lower() for path in paths)
    assert not any(".env" in path.lower() for path in paths)


def test_reference_exact_ell_preserves_signed_values() -> None:
    matrix = torch.tensor(
        [[0.0, -2.0, 3.0, 0.0], [-4.0, 0.0, -5.0, 6.0]],
        dtype=torch.bfloat16,
    )
    values, columns, counts = CORE.pack_exact_ell_reference(matrix, torch=torch)
    contract = CORE.validate_exact_ell(matrix, values, columns, counts, torch=torch)
    assert counts.tolist() == [2, 3]
    assert values[0, :2].tolist() == [-2.0, 3.0]
    assert columns[0, :2].tolist() == [1, 2]
    assert contract["signed_round_trip_exact"] is True
    assert contract["dropped_values"] == 0


def test_attention_opportunity_keeps_causal_integer_denominator() -> None:
    accumulator = CORE.AttentionOpportunityAccumulator(sequence_length=3)
    query = torch.tensor([[[[0.0], [1.0], [0.0]]]])
    value = torch.tensor([[[[1.0], [0.0], [0.0]]]])
    accumulator.update(query, value, torch=torch)
    result = accumulator.finalize()
    assert result["q_only_causal_zero_products"] == 4
    assert result["v_only_causal_zero_products"] == 3
    assert result["causal_products_per_operation"] == 6
    assert result["physical_full_gemm_products_per_operation"] == 9
    assert result["integer_pooling"] is True


def test_installation_still_replaces_only_declared_linears() -> None:
    layer = SimpleNamespace(
        attention=SimpleNamespace(
            query_key_value=torch.nn.Linear(4, 12),
            dense=torch.nn.Linear(4, 4),
        ),
        mlp=SimpleNamespace(
            dense_h_to_4h=torch.nn.Linear(4, 16),
            dense_4h_to_h=torch.nn.Linear(16, 4),
        ),
    )
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[layer]))
    original_qkv = layer.attention.query_key_value
    adapters = SPARSE.install_sparse_linears(model, ["mlp_w2"], torch=torch)
    assert list(adapters) == ["layer_0.mlp_w2"]
    assert isinstance(layer.mlp.dense_4h_to_h, SPARSE.ExactEllLinear)
    assert layer.attention.query_key_value is original_qkv


def test_batch32_requires_calibration_and_is_not_primary() -> None:
    worker = (RUN_DIR / "05_start_worker.sh").read_text(encoding="utf-8")
    calibration = (RUN_DIR / "08_calibrate_batch.py").read_text(encoding="utf-8")
    benchmark = (RUN_DIR / "03_benchmark.py").read_text(encoding="utf-8")
    assert "batch32_memory_calibration" in worker
    assert "minimum_headroom_fraction" in calibration
    assert 'condition_by_id(config, "a4-ol1-kappa-0p5")' in calibration
    assert "two BF16 models; native, adapter-dense, and sparse full forwards" in calibration
    assert 'INCLUDE_BATCH32 = "--include-batch32" in sys.argv' in benchmark
    assert 'config["measurement"]["full_model_batches"] = [1, 32]' in benchmark


def test_occupancy_accumulators_use_410m_widths_and_24_layers() -> None:
    benchmark = load_script("03_benchmark.py")
    config = RUN_CONFIG.load_config()
    condition = RUN_CONFIG.condition_by_id(config, "a7-ol1-kappa-0p5")
    accumulators = benchmark.make_accumulators(condition, config)
    assert len(accumulators) == 7 * 24
    assert accumulators["h.layer_0"].width == 4096
    assert accumulators["a.layer_23"].width == 1024
    assert accumulators["q_post.layer_0"].width == 64
    assert accumulators["q_post.layer_0"].tile_width == 64


def test_verifier_uses_dynamic_batch_decision_and_410m_attention_shape() -> None:
    verifier = (RUN_DIR / "04_verify.py").read_text(encoding="utf-8")
    assert 'decision.get("primary_batch") == 1' in verifier
    assert 'config["measurement"]["full_model_batches"] = [1, 32]' in verifier
    assert '"H": architecture["attention_heads"]' in verifier
    assert 'expected_validation["evaluated_tokens"] * heads' in verifier
    assert 'paired["median"] > 1 and paired["p10"] > 1' in verifier


def test_shell_and_documented_scope_are_fail_closed() -> None:
    for name in ("00_setup_remote.sh", "05_start_worker.sh"):
        text = (RUN_DIR / name).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail")
        assert "+  --" not in text
        assert "+    --" not in text
    readme = (RUN_DIR / "README.md").read_text(encoding="utf-8")
    assert "Sakana-derived" in readme
    assert "batch one" in readme
    assert "Batch 32" in readme
    assert "The worse 410M loss limits any quality-speed Pareto claim" in readme
    assert "not a runtime estimate" in readme


def test_patch_remains_the_hash_pinned_run023_derivative() -> None:
    config = RUN_CONFIG.load_config()
    patch_path = RUN_CONFIG.repo_path(config["upstream"]["patch"])
    assert RUN_CONFIG.sha256_file(patch_path) == config["upstream"]["patch_sha256"]
    patch = patch_path.read_text(encoding="utf-8")
    assert "dense_to_ell_exact_kernel" in patch
    assert "ell_spmm_raw_out" in patch
    assert "__bfloat162float(value) != 0.0f" in patch


def test_invalid_scientific_scope_is_rejected() -> None:
    config = RUN_CONFIG.load_config()
    config["model"]["architecture"]["hidden_size"] = 512
    with pytest.raises(ValueError, match="fixed Pythia-410M"):
        RUN_CONFIG.validate_config(config)
