"""Integer reach accounting for the five paper topologies."""

import pytest
from sparsity_research.ceilings import architecture_ceiling

ARCHITECTURE = dict(
    layers=6, hidden_size=128, ffn_size=512, sequence_length=2048, vocabulary_size=50304
)


@pytest.mark.parametrize(
    "topology,reachable",
    [
        ("A0", ()),
        ("A1-H", ("mlp_w2",)),
        ("HZ", ("attention_output_projection", "mlp_w2")),
        ("A4-Z", ("qkv_projection", "attention_output_projection", "mlp_w1", "mlp_w2")),
        (
            "A7-Z-POST",
            (
                "qkv_projection",
                "qk_scores",
                "probability_value",
                "attention_output_projection",
                "mlp_w1",
                "mlp_w2",
            ),
        ),
    ],
)
def test_paper_reach_uses_integer_counts_once(topology, reachable):
    result = architecture_ceiling(topology, **ARCHITECTURE)
    assert result["reachable_operations"] == list(reachable)
    assert result["reachable_product_count"] == 6 * sum(
        result["per_block_operation_products"][op] for op in reachable
    )
    assert result["model_product_count"] == 18_825_609_216
    assert result["R_model_max_fraction"] == pytest.approx(
        result["reachable_product_count"] / 18_825_609_216
    )


def test_all_sites_reach_block_products_but_exclude_dense_vocabulary():
    result = architecture_ceiling("A7-Z-POST", **ARCHITECTURE)
    assert (
        result["reachable_product_count"]
        == result["block_product_count"]
        == 5_638_717_440
    )
    assert result["R_model_max_fraction"] == pytest.approx(0.2995237697384911)
    assert result["lm_head_product_count"] == 2048 * 128 * 50304


def test_two_site_context_gate_does_not_reach_attention_scores_or_values():
    result = architecture_ceiling("HZ", **ARCHITECTURE)
    assert result["reachable_product_count"] == 6 * 2048 * (512 * 128 + 128 * 128)


def test_context_length_changes_attention_reach_without_quadratic_projection_counts():
    short = architecture_ceiling("A7-Z-POST", **ARCHITECTURE)
    long = architecture_ceiling(
        "A7-Z-POST", **{**ARCHITECTURE, "sequence_length": 4096}
    )
    assert (
        long["per_block_operation_products"]["mlp_w2"]
        == 2 * short["per_block_operation_products"]["mlp_w2"]
    )
    assert long["R_model_max_fraction"] > short["R_model_max_fraction"]
