"""Guard cohort identity, count pooling, regression and matched GM estimands."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernels", HERE / "07_kernel_realization.py")
kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernels)


def test_complete_qualified_cohort_and_canonical_counts():
    data = kernels.read_evidence()
    assert data == json.loads((HERE / "data/kernel-realization.json").read_text(encoding="utf-8"))
    assert len(data["points"]) == 90
    assert data["coverage"] == dict(documents=500, sequences=338, input_tokens=692224,
                                    prediction_tokens=691886, excluded_tail_tokens=1444, seed_count=1)
    assert len(set(data["timing_block_indices"])) == 64
    for candidate in kernels.CANDIDATES:
        points = [p for p in data["points"] if p["candidate"] == candidate]
        assert {p["condition"] for p in points} == {f"c{i:02d}" for i in range(1, 31)}
        assert len({p["evidence_id"] for p in points}) == 30
        assert all(p["qualified"] and len(p["replicates"]) == 3 for p in points)
        for p in points:
            counts = p["canonical_counts"]
            zeros = sum(op["zero_product_count"] for op in counts["per_operation"].values())
            assert p["sparsity_percent"] == pytest.approx(100 * zeros / counts["model_product_count"])
    endpoint, = [p for p in data["points"] if p["candidate"] == "k050" and p["condition"] == "c30"]
    assert (endpoint["family"], endpoint["dose"]) == ("A7-OL1", .5)
    assert endpoint["sparsity_percent"] == pytest.approx(27.482684296304843)
    assert endpoint["speedup"] == pytest.approx(1.7831745639788332)


def test_unweighted_ols_has_intercept_and_uses_all_30_points():
    data = kernels.read_evidence()
    points = [p for p in data["points"] if p["candidate"] == "k050"]
    x = np.array([p["sparsity_percent"] for p in points])
    y = np.array([p["speedup"] for p in points])
    design = np.column_stack([np.ones(30), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fit = data["regression"]
    assert [fit["intercept"], fit["slope_per_percentage_point"]] == pytest.approx(coefficients)
    assert fit["intercept"] == pytest.approx(.9512951279669726)
    assert fit["r2"] == pytest.approx(.8167176697920289)
    fig = kernels.make_figure(data)
    assert len(fig.axes) == 2
    plotted = np.concatenate([c.get_offsets() for c in fig.axes[0].collections])
    assert sorted(map(tuple, plotted)) == sorted(zip(x, y))
    assert {c.get_gid() for c in fig.axes[1].collections} == set(kernels.CANDIDATES)
    kernels.plt.close(fig)


def test_ablation_ratios_are_matched_multiplicative_changes():
    data = kernels.read_evidence()
    expected = [1.182857976961071, 1.2505587528850488, 1.2339917065809438]
    assert [r["geomean_speedup"] for r in data["ablations"]] == pytest.approx(expected)
    for ref, trt, increment in zip(expected[:-1], expected[1:], data["increments"]):
        ratios = np.array([p["ratio"] for p in increment["pairs"]])
        assert len(ratios) == 30
        assert np.exp(np.log(ratios).mean()) == pytest.approx(trt / ref)
        assert increment["relative_change_percent"] == pytest.approx(100 * (trt / ref - 1))
        assert not np.isclose(increment["relative_change_percent"], 100 * (trt - ref))
    assert [f'{r["relative_change_percent"]:+.1f}%' for r in data["increments"]] == ["+5.7%", "-1.3%"]
    assert all(p["ratio"] < 1 for p in data["increments"][1]["pairs"])
