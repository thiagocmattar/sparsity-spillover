from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM

from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels"


def load_module(alias: str, path: Path):
    specification = importlib.util.spec_from_file_location(alias, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[alias] = module
    specification.loader.exec_module(module)
    return module


RUN_CONFIG = load_module("run023_run_config", RUN_DIR / "run_config.py")
CORE = load_module("run023_benchmark_core", RUN_DIR / "benchmark_core.py")
SPARSE = load_module("run023_pythia_sparse", RUN_DIR / "pythia_sparse.py")


def load_script(name: str):
    path = RUN_DIR / name
    specification = importlib.util.spec_from_file_location(f"run023_{path.stem}", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    previous = {
        name: sys.modules.get(name)
        for name in ("run_config", "benchmark_core", "pythia_sparse")
    }
    sys.modules["run_config"] = RUN_CONFIG
    sys.modules["benchmark_core"] = CORE
    sys.modules["pythia_sparse"] = SPARSE
    try:
        specification.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


def test_config_is_exact_twelve_condition_partition() -> None:
    config = RUN_CONFIG.load_config()
    assert tuple(condition["id"] for condition in config["conditions"]) == RUN_CONFIG.EXPECTED_IDS
    assert tuple(config["condition_partition"]["sentinel"]) == RUN_CONFIG.EXPECTED_SENTINELS
    assert len(config["condition_partition"]["remainder"]) == 6
    assert not set(config["condition_partition"]["sentinel"]) & set(
        config["condition_partition"]["remainder"]
    )
    assert config["model"]["architecture"] == {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "layers": 6,
        "attention_heads": 8,
        "head_size": 64,
        "sequence_length": 2048,
        "vocab_size": 50304,
        "attention_bias": True,
        "mlp_bias": True,
    }
    assert config["validation"]["source_reproduction_batch_size"] == 4
    assert config["validation"]["runtime_equivalence_batch_size"] == 32
    assert config["validation"]["logical_opportunity_batch_size"] == 1


def test_transfer_allowlist_is_phase_specific_and_excludes_training_state() -> None:
    config = RUN_CONFIG.load_config()
    prepare = load_script("00_prepare_inputs.py")
    sentinel = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in prepare.transfer_files(config, "sentinel")
    ]
    remainder = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in prepare.transfer_files(config, "remainder")
    ]
    assert sum(path.endswith("model.safetensors") for path in sentinel) == 6
    assert sum(path.endswith("model.safetensors") for path in remainder) == 6
    assert set(sentinel) != set(remainder)
    for paths in (sentinel, remainder):
        assert any(path.endswith("validation/tokens.int32.bin") for path in paths)
        assert any(path.endswith("sakana-pythia70.patch") for path in paths)
        assert not any("training_state.pt" in path for path in paths)
        assert not any("/train/" in path for path in paths)
        assert not any(".env" in path.lower() for path in paths)


def test_reference_exact_ell_preserves_signed_values_and_order() -> None:
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
    assert "indices.to(torch.long)[valid]" in (RUN_DIR / "benchmark_core.py").read_text(encoding="utf-8")
    assert "indices.to(torch.int32)[valid]" in (RUN_DIR / "02_remote_preflight.py").read_text(encoding="utf-8")


def test_attention_opportunity_uses_causal_integer_weights() -> None:
    accumulator = CORE.AttentionOpportunityAccumulator(sequence_length=3)
    query = torch.tensor([[[[0.0], [1.0], [0.0]]]])
    value = torch.tensor([[[[1.0], [0.0], [0.0]]]])
    accumulator.update(query, value, torch=torch)
    result = accumulator.finalize()
    assert result["q_only_causal_zero_products"] == 4
    assert result["v_only_causal_zero_products"] == 3
    assert result["q_only_physical_full_gemm_zero_products"] == 6
    assert result["v_only_physical_full_gemm_zero_products"] == 6
    assert result["causal_products_per_operation"] == 6
    assert result["physical_full_gemm_products_per_operation"] == 9
    assert result["integer_pooling"] is True


def test_covered_opportunity_is_subset_with_unchanged_denominator() -> None:
    logical = {
        "measured": {
            "R_model": 0.2,
            "block_zero_product_count": 200,
            "model_product_count": 1000,
            "per_operation": {
                "mlp_w2": {"zero_product_count": 100},
                "qk_scores": {"zero_product_count": 50},
                "probability_value": {"zero_product_count": 30},
            },
        }
    }
    attention = {
        "q_only_causal_zero_products": 20,
        "v_only_causal_zero_products": 10,
    }
    result = CORE.covered_opportunity(logical, ["mlp_w2"], attention=attention)
    assert result["covered_zero_product_count"] == 130
    assert result["canonical_model_product_count"] == 1000
    assert result["R_covered"] == pytest.approx(0.13)
    assert result["canonical_R_model"] == pytest.approx(0.2)


def test_full_model_and_separate_attention_coverage_are_not_conflated() -> None:
    benchmark = (RUN_DIR / "03_benchmark.py").read_text(encoding="utf-8")
    assert '"kernel_covered_opportunity": full_model_coverage' in benchmark
    assert '"kernel_covered_opportunity_with_separate_attention": linear_plus_attention_coverage' in benchmark
    assert '"full_model_r_covered_excludes_separately_timed_attention": True' in benchmark


def test_pv_transpose_identity_is_exact_in_fp32() -> None:
    generator = torch.Generator().manual_seed(23)
    probabilities = torch.rand((2, 4, 4), generator=generator)
    values = torch.rand((2, 4, 3), generator=generator)
    expected = probabilities @ values
    actual = CORE.pv_v_left_equivalent(probabilities, values)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_workspace_reuses_shape_and_adapter_dense_preserves_bias() -> None:
    workspace = SPARSE.ExactEllWorkspace()
    workspace.ensure(2, 4, 8, device=torch.device("cpu"), torch=torch)
    first_values = workspace.values
    workspace.ensure(2, 4, 8, device=torch.device("cpu"), torch=torch)
    assert workspace.allocations == 1
    assert workspace.values is first_values
    workspace.ensure(3, 4, 8, device=torch.device("cpu"), torch=torch)
    assert workspace.allocations == 2
    linear = torch.nn.Linear(4, 8)
    adapter = SPARSE.ExactEllLinear(linear, operation="test", torch=torch)
    adapter.mode = "dense"
    inputs = torch.randn(2, 4)
    assert torch.equal(adapter(inputs), linear(inputs))
    adapter.mode = "sparse"
    with torch.inference_mode(), pytest.raises(TypeError, match="CUDA BF16"):
        adapter(inputs)


def test_installation_replaces_only_declared_linears() -> None:
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


def test_dense_adapters_preserve_real_a7_checkpoint_and_gate_order() -> None:
    config = RUN_CONFIG.load_config()
    condition = RUN_CONFIG.condition_by_id(config, "a7-ol1-kappa-0")
    model = load_checkpoint_pythia(
        AutoModelForCausalLM,
        RUN_CONFIG.checkpoint_path(config, condition),
        torch=torch,
    ).eval()
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        before = model(input_ids=input_ids, use_cache=False).logits
    adapters = SPARSE.install_sparse_linears(model, condition["linear_operations"], torch=torch)
    SPARSE.set_adapter_mode(adapters, "dense")
    model.eval()
    with torch.inference_mode():
        after = model(input_ids=input_ids, use_cache=False).logits
    assert torch.equal(before, after)
    assert topology_metadata(model)["active_sites"] == condition["active_sites"]


def test_patch_is_pinned_and_contains_narrow_width_and_signed_pack_fixes() -> None:
    config = RUN_CONFIG.load_config()
    patch_path = RUN_CONFIG.repo_path(config["upstream"]["patch"])
    assert RUN_CONFIG.sha256_file(patch_path) == config["upstream"]["patch_sha256"]
    text = patch_path.read_text(encoding="utf-8")
    assert "for (int n_base = 0; n_base < N_cols" in text
    assert "const bool output_active = n_out < N_cols" in text
    assert "dense_to_ell_exact_kernel" in text
    assert "__ballot_sync(0xffffffffu, nonzero)" in text
    assert "__bfloat162float(value) != 0.0f" in text
    assert "dense_to_ell_exact_out" in text
    assert "ell_spmm_raw_out" in text


def test_static_preflight_validates_all_real_checkpoints_and_cache() -> None:
    module = load_script("01_static_preflight.py")
    config = RUN_CONFIG.load_config()
    records = [module.validate_condition(config, condition) for condition in config["conditions"]]
    assert len(records) == 12
    assert all(all(record["assertions"].values()) for record in records)
    assert all(len(record["linear_tensor_shapes"]) == 48 for record in records)
    assert all(module.validate_validation(config)["assertions"].values())


def test_documented_scope_does_not_claim_official_attention_support() -> None:
    readme = (RUN_DIR / "README.md").read_text(encoding="utf-8")
    assert "Sakana-derived" in readme
    assert "not a drop-in causal-attention implementation" in readme
    assert "leaves attention dense in the full-model sparse-linear path" in readme
    assert "post-hoc TEAL frontiers are deliberately deferred" in readme
    assert "not launch authorization" in readme


def test_shell_sources_have_no_collapsed_patch_continuations() -> None:
    for name in ("00_setup_remote.sh", "05_start_worker.sh"):
        text = (RUN_DIR / name).read_text(encoding="utf-8")
        assert "+  --" not in text
        assert "+    --" not in text
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail")


def test_monitor_distinguishes_stage_from_phase_etc_and_retains_guard_cost() -> None:
    benchmark = (RUN_DIR / "03_benchmark.py").read_text(encoding="utf-8")
    monitor = (RUN_DIR / "06_monitor.py").read_text(encoding="utf-8")
    assert "phase_projected_completion_unix" in benchmark
    assert 'etc_scope = "current_stage_only"' in monitor
    assert '"phase_projection"' in monitor
    assert '"phase_projection_overdue"' in monitor
    assert '"guard_remaining_cost": guard_remaining_cost' in monitor
