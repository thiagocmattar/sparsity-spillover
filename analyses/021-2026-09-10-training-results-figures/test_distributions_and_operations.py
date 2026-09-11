"""Check probability mass, operation units, and the two-checkpoint evidence join."""

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("composite", HERE / "04_distributions_and_operations.py")
composite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composite)


def test_density_preserves_zero_and_tail_mass_without_renormalizing():
    group = {"histogram": [2] * 10 + [1] * 10, "total": 100,
             "exact_zero_count": 50, "underflow": 3, "overflow": 17, "nonfinite": 0}
    grid = {"lower": -1., "upper": 1., "bins": 20}
    edges, counts, density = composite.rebin_density(group, grid)
    np.testing.assert_array_equal(counts, [20, 10])
    np.testing.assert_allclose(edges, [-1., 0., 1.])
    np.testing.assert_allclose(density, [.2, .1])
    assert np.isclose(np.sum(density * np.diff(edges)) + .50 + .03 + .17, 1.)


def test_contributions_use_common_model_denominator_including_output_projection():
    ops = {name: {"product_count": total, "zero_product_count": zero}
           for (name, _, _), total, zero in zip(composite.OPERATIONS,
               [10, 20, 30, 40, 50, 60], [1, 2, 6, 8, 20, 30])}
    counts = {"per_operation": ops, "block_product_count": 210,
              "block_zero_product_count": 67, "lm_head_product_count": 90,
              "model_product_count": 300}
    values = composite.operation_contributions(counts)
    assert np.isclose(sum(values.values()), 100 * 67 / 300)
    assert np.isclose(values["qk_scores"], 100 * 20 / 300)
    assert np.isclose(values["probability_value"], 100 * 30 / 300)


def test_retained_sources_reproduce_reversal_and_all_plotted_counts():
    data = composite.read_evidence()
    assert data == json.loads((HERE / "data/14m-distributions-and-operations.json").read_text())
    assert data["scale"] == "14M" and data["kappa"] == .5
    assert data["coverage"] == dict(documents=500, sequences=338, input_tokens=692224,
                                    prediction_tokens=691886, excluded_tail_tokens=1444, seed_count=1)
    a4, a7 = data["records"]
    assert [r["id"] for r in data["records"]] == ["14M:A4-OL1:0.5", "14M:A7-OL1:0.5"]
    for row, zeros, total, attention, loss in zip(data["records"],
            [(99.31, .22), (93.64, 95.60)], [12.71, 27.48], [.06, 16.77], [6.0380, 5.8294]):
        np.testing.assert_allclose([row["groups"][name]["exact_zero_percent"] for name in composite.GROUPS],
                                   zeros, rtol=0, atol=.005)
        assert abs(row["S_model_percent"] - total) < .005
        assert abs(row["attention_contribution_pp"] - attention) < .005
        assert abs(row["loss"] - loss) < .00005
        for group in row["groups"].values():
            assert sum(group["bin_counts"]) + group["exact_zero_count"] + group["underflow"] + group["overflow"] == group["total"]
    assert a4["groups"]["FFN activations"]["exact_zero_percent"] > a7["groups"]["FFN activations"]["exact_zero_percent"]
    assert a4["S_model_percent"] < a7["S_model_percent"] and a4["loss"] > a7["loss"]
