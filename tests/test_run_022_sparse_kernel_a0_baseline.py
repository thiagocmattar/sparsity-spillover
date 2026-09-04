from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "022-2026-09-04-pythia14m-sparse-kernel-a0-baseline"


def load_module(alias: str, path: Path):
    specification = importlib.util.spec_from_file_location(alias, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[alias] = module
    specification.loader.exec_module(module)
    return module


RUN_CONFIG = load_module("run022_run_config", RUN_DIR / "run_config.py")
BENCHMARK_CORE = load_module("run022_benchmark_core", RUN_DIR / "benchmark_core.py")
OccupancyAccumulator = BENCHMARK_CORE.OccupancyAccumulator
histogram_quantile = BENCHMARK_CORE.histogram_quantile
pack_exact_ell = BENCHMARK_CORE.pack_exact_ell
unpack_exact_ell = BENCHMARK_CORE.unpack_exact_ell
validate_raw_ell_contract = BENCHMARK_CORE.validate_raw_ell_contract
load_config = RUN_CONFIG.load_config


def load_script(name: str):
    path = RUN_DIR / name
    specification = importlib.util.spec_from_file_location(f"run022_{path.stem}", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    previous_config = sys.modules.get("run_config")
    previous_core = sys.modules.get("benchmark_core")
    sys.modules["run_config"] = RUN_CONFIG
    sys.modules["benchmark_core"] = BENCHMARK_CORE
    try:
        specification.loader.exec_module(module)
    finally:
        if previous_config is None:
            sys.modules.pop("run_config", None)
        else:
            sys.modules["run_config"] = previous_config
        if previous_core is None:
            sys.modules.pop("benchmark_core", None)
        else:
            sys.modules["benchmark_core"] = previous_core
    return module


def test_exact_ell_round_trip_and_contract() -> None:
    matrix = torch.tensor(
        [[0.0, 1.0, -2.0, 0.0, 3.0], [4.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    values, indices, counts, stride = pack_exact_ell(matrix, torch=torch, alignment=4)
    assert stride == 4
    assert counts.tolist() == [3, 1]
    assert torch.equal(unpack_exact_ell(values, indices, counts, columns=5, torch=torch), matrix)
    rhs = torch.ones((5, 8), dtype=torch.bfloat16)
    contract = validate_raw_ell_contract(
        values,
        indices,
        counts,
        rhs,
        overflow_threshold=stride,
        require_cuda=False,
        torch=torch,
    )
    assert contract == {"M": 2, "K": 5, "N": 8, "stride": 4, "max_row_nnz": 3}


def test_raw_ell_contract_rejects_silent_overflow() -> None:
    matrix = torch.ones((2, 5), dtype=torch.bfloat16)
    values, indices, counts, stride = pack_exact_ell(matrix, torch=torch, alignment=1)
    with pytest.raises(ValueError, match="silently skip"):
        validate_raw_ell_contract(
            values,
            indices,
            counts,
            torch.ones((5, 8), dtype=torch.bfloat16),
            overflow_threshold=stride - 1,
            require_cuda=False,
            torch=torch,
        )


def test_raw_ell_contract_rejects_output_width_not_divisible_by_eight() -> None:
    matrix = torch.ones((2, 5), dtype=torch.bfloat16)
    values, indices, counts, stride = pack_exact_ell(matrix, torch=torch, alignment=1)
    with pytest.raises(ValueError, match="divisible by 8"):
        validate_raw_ell_contract(
            values,
            indices,
            counts,
            torch.ones((5, 7), dtype=torch.bfloat16),
            overflow_threshold=stride,
            require_cuda=False,
            torch=torch,
        )


def test_occupancy_pools_integer_counts_before_division() -> None:
    accumulator = OccupancyAccumulator(
        width=4,
        tile_width=2,
        near_zero_thresholds=(0.001, 0.01),
        payload_capacities={"one": 1},
    )
    accumulator.update(
        torch.tensor([[0.0, 1.0, 0.0005, 2.0], [-0.01, 0.0, 3.0, 0.0]]),
        torch=torch,
    )
    result = accumulator.finalize()
    assert result["elements"] == 8
    assert result["exact_zero_count"] == 3
    assert result["exact_zero_mass"] == pytest.approx(3 / 8)
    assert result["near_zero"]["0.001"]["count"] == 4
    assert result["near_zero"]["0.01"]["count"] == 5
    assert result["row_nnz"]["histogram"][2:4] == [1, 1]
    assert result["tile_nnz"]["histogram"][1:3] == [3, 1]
    assert result["tile_overflow"]["one"]["count"] == 1
    assert result["integer_pooling"] is True


def test_histogram_quantile_uses_nearest_rank() -> None:
    histogram = [0, 1, 2, 1]
    assert histogram_quantile(histogram, 0.0) == 1
    assert histogram_quantile(histogram, 0.5) == 2
    assert histogram_quantile(histogram, 1.0) == 3


def test_relative_error_rejects_nonfinite_output() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        BENCHMARK_CORE.relative_errors(
            torch.tensor([float("nan")]), torch.tensor([1.0]), torch=torch
        )


def test_config_and_transfer_allowlist_preserve_scope() -> None:
    config = load_config()
    assert config["model"]["topology_id"] == "A0"
    assert config["validation"]["complete_blocks"] == 338
    assert config["measurement"]["raw_ell_rows"] == [256, 2048]
    prepare = load_script("00_prepare_inputs.py")
    relative = [path.relative_to(REPO_ROOT).as_posix() for path in prepare.transfer_files(config)]
    assert any(path.endswith("model.safetensors") for path in relative)
    assert any(path.endswith("validation/tokens.int32.bin") for path in relative)
    assert not any("training_state.pt" in path for path in relative)
    assert not any("/train/" in path for path in relative)
    assert not any(".env" in path.lower() for path in relative)


def test_static_preflight_validates_real_checkpoint_and_cache() -> None:
    module = load_script("01_static_preflight.py")
    config = load_config()
    checkpoint = module.validate_checkpoint(config)
    validation = module.validate_validation(config)
    assert all(checkpoint["assertions"].values())
    assert len(checkpoint["mlp_shapes"]) == 24
    assert all(validation["assertions"].values())


def test_followup_plan_does_not_claim_released_attention_support() -> None:
    text = (RUN_DIR / "FOLLOWUP_PLAN.md").read_text(encoding="utf-8")
    assert "released TwELL MLP converter does not implement attention" in text
    assert "60 configurations" in text
    assert "24 checkpoints" in text
