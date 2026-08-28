import pytest

from sparsity_research.ceilings import architecture_ceiling


PYTHIA_14M = {
    "layers": 6,
    "hidden_size": 128,
    "ffn_size": 512,
    "sequence_length": 2048,
    "vocabulary_size": 50_304,
}

PYTHIA_12B = {
    "layers": 36,
    "hidden_size": 5_120,
    "ffn_size": 20_480,
    "sequence_length": 2048,
    "vocabulary_size": 50_688,
}


@pytest.mark.parametrize(
    ("topology", "expected_14m", "expected_12b"),
    [
        ("A0", 0.0000, 0.0000),
        ("A1-H", 4.2777, 31.5577),
        ("A3", 11.7637, 86.7837),
        ("A4-Q", 20.3233, 88.3623),
        ("A4-K", 20.3233, 88.3623),
        ("A4-V", 21.3928, 96.2518),
        ("A5-QK-PRE", 20.3233, 88.3623),
        ("A5-QK-POST", 20.3233, 88.3623),
        ("A6-PRE", 29.9524, 97.8304),
        ("A6-POST", 29.9524, 97.8304),
    ],
)
def test_reproduces_manuscript_topology_table(topology, expected_14m, expected_12b):
    assert architecture_ceiling(topology, **PYTHIA_14M)[
        "R_model_max_percent"
    ] == pytest.approx(expected_14m, abs=5e-5)
    assert architecture_ceiling(topology, **PYTHIA_12B)[
        "R_model_max_percent"
    ] == pytest.approx(expected_12b, abs=5e-5)


def test_a2_reaches_only_both_mlp_projections():
    result = architecture_ceiling("A2", **PYTHIA_14M)
    assert result["reachable_operations"] == ["mlp_w1", "mlp_w2"]
    assert result["reachable_product_count"] == 6 * 2048 * 8 * 128 * 128


def test_q_and_k_are_one_reachable_qk_operation_not_double_counted():
    q = architecture_ceiling("A4-Q", **PYTHIA_14M)
    k = architecture_ceiling("A4-K", **PYTHIA_14M)
    both = architecture_ceiling("A5-QK-POST", **PYTHIA_14M)
    assert q["reachable_product_count"] == k["reachable_product_count"]
    assert q["reachable_product_count"] == both["reachable_product_count"]


def test_v_reaches_pv_and_context_output_projection():
    result = architecture_ceiling("A4-V", **PYTHIA_14M)
    assert "probability_value" in result["reachable_operations"]
    assert "attention_output_projection" in result["reachable_operations"]


def test_a6_reaches_every_block_operation_but_not_lm_head():
    result = architecture_ceiling("A6-POST", **PYTHIA_14M)
    assert result["reachable_product_count"] == result["block_product_count"]
    assert result["R_model_max_fraction"] < 1.0


def test_ceiling_changes_with_sequence_length():
    long_context = {**PYTHIA_12B, "sequence_length": 50_000}
    result = architecture_ceiling("A3", **long_context)
    assert result["R_model_max_percent"] == pytest.approx(49.9, abs=0.1)
