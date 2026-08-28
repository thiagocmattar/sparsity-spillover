import pytest
import torch

from sparsity_research.metrics import (
    ActivationAccumulator,
    LOGICAL_OPERATIONS,
    linear_zero_product_counts,
    pool_weight_norm,
    pv_zero_product_counts,
    qk_zero_product_counts,
    summarize_logical_products,
    weight_statistics,
)


def test_activation_accumulator_pools_counts_and_moments_before_dividing():
    accumulator = ActivationAccumulator((0.0, 0.5))
    accumulator.update(
        {
            "h.layer_0": torch.tensor([0.0, 1.0]),
            "h.layer_1": torch.tensor([0.25, 2.0, float("nan")]),
        },
        torch=torch,
    )
    pooled = accumulator.pooled_by_site()[0]
    assert pooled["name"] == "h"
    assert pooled["total"] == 5
    assert pooled["finite"] == 4
    assert pooled["exact_zero_count"] == 1
    assert pooled["threshold_hits"] == {"0": 1, "0.5": 2}
    assert pooled["rms"] == pytest.approx((5.0625 / 4) ** 0.5)
    assert pooled["l2_norm"] == pytest.approx(5.0625**0.5)


def test_weight_norms_keep_names_and_pool_squared_norms():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        model[1].weight.copy_(torch.tensor([[0.0, 12.0]]))
    rows = weight_statistics(model)
    pooled = pool_weight_norm(rows)
    assert {row["name"] for row in rows} == {"0.weight", "1.weight"}
    assert pooled["l2_norm"] == 13.0


def test_weight_statistics_names_pythia_operation_role_and_layer():
    model = torch.nn.Module()
    model.gpt_neox = torch.nn.Module()
    layer = torch.nn.Module()
    layer.attention = torch.nn.Module()
    layer.attention.query_key_value = torch.nn.Linear(2, 6, bias=False)
    model.gpt_neox.layers = torch.nn.ModuleList([layer])
    row = weight_statistics(model)[0]
    assert row["layer"] == 0
    assert row["role"] == "qkv_projection"


def test_linear_zero_product_count_expands_each_zero_over_outputs():
    zero, total = linear_zero_product_counts(torch.tensor([[0.0, 1.0]]), output_features=3, torch=torch)
    assert (zero, total) == (3, 6)


def test_qk_counts_only_valid_causal_pairs():
    query = torch.tensor([[[[0.0], [1.0]]]])
    key = torch.tensor([[[[1.0], [0.0]]]])
    zero, total = qk_zero_product_counts(query, key, torch=torch)
    assert total == 3
    assert zero == 2


def test_pv_counts_probability_or_value_zero_operands():
    probabilities = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    value = torch.tensor([[[[1.0], [0.0]]]])
    zero, total = pv_zero_product_counts(probabilities, value, torch=torch)
    assert total == 3
    assert zero == 1


def test_measured_denominators_are_distinct_and_do_not_infer_analytic_ceiling():
    zero = {name: 1 for name in LOGICAL_OPERATIONS}
    total = {name: 10 for name in LOGICAL_OPERATIONS}
    result = summarize_logical_products(zero, total, lm_head_product_count=40)
    assert result["R_block"] == pytest.approx(0.1)
    assert result["R_model"] == pytest.approx(0.06)
    assert "R_max" not in result and "R_model_max" not in result
