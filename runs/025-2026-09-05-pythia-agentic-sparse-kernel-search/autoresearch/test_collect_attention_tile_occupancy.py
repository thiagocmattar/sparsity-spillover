import importlib.util
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run025_attention_occupancy", HERE / "collect_attention_tile_occupancy.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_temporal_tiles_are_counted_from_consecutive_rows_not_marginals():
    value = torch.ones(1, 2, 64, 32, dtype=torch.bfloat16)
    value[:, 0, :16] = 0
    value[:, 1, ::2] = 0

    row = MODULE.attention_counts(value)

    assert row["zero_rows"] == 16 + 32
    assert row["temporal_tiles"]["t16"]["zero_tiles"] == 1
    assert row["temporal_tiles"]["t32"]["zero_tiles"] == 0
    assert row["temporal_tiles"]["t1"]["zero_tiles"] == 48


def test_integer_pooling_precedes_fraction_calculation():
    first = MODULE.attention_counts(
        torch.zeros(1, 2, 64, 32, dtype=torch.bfloat16)
    )
    second = MODULE.attention_counts(
        torch.ones(1, 2, 64, 32, dtype=torch.bfloat16)
    )
    pooled = {}
    MODULE.add_counts(pooled, first)
    MODULE.add_counts(pooled, second)
    result = MODULE.with_fractions(pooled)

    assert result["zero_fraction"] == 0.5
    assert result["zero_row_fraction"] == 0.5
    assert result["temporal_tiles"]["t16"]["zero_tile_fraction"] == 0.5
    assert result["calls"] == 2


def test_mismatched_shape_pooling_is_rejected():
    destination = MODULE.attention_counts(
        torch.zeros(1, 2, 64, 32, dtype=torch.bfloat16)
    )
    source = MODULE.attention_counts(
        torch.zeros(1, 2, 64, 64, dtype=torch.bfloat16)
    )

    try:
        MODULE.add_counts(destination, source)
    except ValueError as error:
        assert "mismatched attention shapes" in str(error)
    else:
        raise AssertionError("Mismatched attention shapes were pooled")
