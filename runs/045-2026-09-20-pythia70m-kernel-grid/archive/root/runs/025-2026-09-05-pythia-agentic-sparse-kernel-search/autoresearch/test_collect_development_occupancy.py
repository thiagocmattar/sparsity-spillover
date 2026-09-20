"""CPU mathematical/counting and hook-identity checks; no GPU performance claim."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("run025_occupancy_tested", Path(__file__).with_name("collect_development_occupancy.py"))
occupancy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(occupancy)


def test_actual_2d_tiles_are_not_inferred_from_scalar_zero_fraction():
    value = torch.zeros(32, 64, dtype=torch.bfloat16)
    value[15, 31] = -2
    row = occupancy.tensor_counts(value)
    assert row["elements"] == row["finite_count"] == 2048
    assert row["zero_count"] == 2047 and row["zero_rows"] == 31
    assert row["tiles"]["m16_k32"] == {"shape": [16, 32], "tiles": 4, "zero_tiles": 3}
    assert row["tiles"]["m32_k32"]["zero_tiles"] == 1
    assert row["tiles"]["m16_k64"]["zero_tiles"] == 1
    dispersed = value.clone()
    dispersed[31, 63] = -2
    row2 = occupancy.tensor_counts(dispersed)
    assert row2["tiles"]["m16_k32"]["zero_tiles"] == 2
    assert row2["tiles"]["m32_k32"]["zero_tiles"] == 0
    assert row2["tiles"]["m16_k64"]["zero_tiles"] == 0


def test_attention_counts_rows_and_whole_head_sequences_without_padding():
    value = torch.zeros(1, 2, 32, 32, dtype=torch.bfloat16)
    value[0, 1, 7, 3] = 1
    row = occupancy.tensor_counts(value, attention=True)
    assert row["head_sequences"] == 2 and row["zero_head_sequences"] == 1
    assert row["rows"] == 64 and row["zero_rows"] == 63
    assert row["tiles"] == {}


def test_integer_first_pooling_handles_unequal_batch_counts():
    first = occupancy.tensor_counts(torch.zeros(32, 64, dtype=torch.bfloat16))
    second = occupancy.tensor_counts(torch.ones(64, 64, dtype=torch.bfloat16))
    pooled = {}
    occupancy.add_counts(pooled, first)
    occupancy.add_counts(pooled, second)
    result = occupancy.with_fractions(pooled)
    assert result["elements"] == 6144 and result["zero_count"] == 2048
    assert result["zero_fraction"] == 1 / 3
    assert result["all_zero_row_fraction"] == 1 / 3
    assert result["tiles"]["m16_k32"]["all_zero_tile_fraction"] == 1 / 3
    assert first["calls"] == second["calls"] == 1


def test_rejects_heldout_endpoints_and_uncovered_tile_shapes():
    manifest = {"checkpoints": [{"id": "14m/a0", "partition": "development"},
                                {"id": "14m/a7-0p1", "partition": "heldout"}]}
    assert occupancy.development_condition(manifest, "14m/a0")["id"] == "14m/a0"
    for identifier in ("missing", "14m/a7-0p1"):
        with pytest.raises(ValueError, match="held-out"):
            occupancy.development_condition(manifest, identifier)
    with pytest.raises(ValueError, match="no padded"):
        occupancy.tensor_counts(torch.zeros(33, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16"):
        occupancy.tensor_counts(torch.zeros(32, 64))


def test_native_pre_hooks_observe_post_gate_inputs_without_mutation():
    mlp = SimpleNamespace(dense_h_to_4h=torch.nn.Identity(), dense_4h_to_h=torch.nn.Identity())
    attention = SimpleNamespace(query_key_value=torch.nn.Identity(), dense=torch.nn.Identity(),
                                q_post_site=torch.nn.Identity(), k_post_site=torch.nn.Identity(), v_site=torch.nn.Identity())
    layer = SimpleNamespace(mlp=mlp, attention=attention)
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[layer]))
    value = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    with occupancy.NativeOccupancy(model) as capture:
        for module in (mlp.dense_h_to_4h, mlp.dense_4h_to_h, attention.query_key_value, attention.dense):
            assert module(value) is value
        heads = value.reshape(1, 1, 32, 64)
        for module in (attention.q_post_site, attention.k_post_site, attention.v_site):
            assert module(heads) is heads
        summary = capture.summary()
        assert set(summary["pooled_by_site"]) == {"a", "m", "h", "z", "q_post", "k_post", "v"}
        assert not summary["missing_attention_taps"]
    assert not capture.handles
    for module in (mlp.dense_h_to_4h, attention.query_key_value, attention.q_post_site):
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_missing_attention_taps_are_reported_and_never_installed():
    mlp = SimpleNamespace(dense_h_to_4h=torch.nn.Identity(), dense_4h_to_h=torch.nn.Identity())
    attention = SimpleNamespace(query_key_value=torch.nn.Identity(), dense=torch.nn.Identity())
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)]))
    with occupancy.NativeOccupancy(model) as capture:
        assert capture.missing_attention_taps == ["q_post.layer_0", "k_post.layer_0", "v.layer_0"]
        assert not hasattr(attention, "q_post_site")


def test_nonfinite_values_remain_counted_and_visible():
    value = torch.zeros(32, 64, dtype=torch.bfloat16)
    value[0, 0] = float("nan")
    row = occupancy.tensor_counts(value)
    assert row["finite_count"] == row["elements"] - 1
    assert row["zero_count"] == row["elements"] - 1
