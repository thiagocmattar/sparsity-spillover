"""Guard cohort identity, both regressions and the projection-gain estimand."""

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
        assert {name: sum(p["visual_family"] == name for p in points) for name in kernels.STYLE} == {
            "Baseline": 1, "1-site": 9, "4-site": 10, "7-site": 10}
        assert all(p["family"] == "A0" for p in points if p["visual_family"] == "Baseline")
        assert all(p["family"].startswith("A1-H") for p in points if p["visual_family"] == "1-site")
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
    fig = kernels.make_figure(data)
    assert len(fig.axes) == 2
    cases = (([p for p in data["points"] if p["candidate"] == "k050"], "sparsity_percent", "speedup", "regression"),
             (data["projection_points"], "bypass_percent", "projection_sparse_gain", "projection_regression"))
    for ax, (points, x_key, y_key, fit_key) in zip(fig.axes, cases):
        x = np.array([p[x_key] for p in points])
        y = np.array([p[y_key] for p in points])
        design = np.column_stack([np.ones(30), x])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        fit = data[fit_key]
        assert [fit["intercept"], fit["slope_per_percentage_point"]] == pytest.approx(coefficients)
        assert fit["r2"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
        plotted = np.concatenate([c.get_offsets() for c in ax.collections])
        assert sorted(map(tuple, plotted)) == sorted(zip(x, y))
        assert {c.get_gid() for c in ax.collections} == set(kernels.STYLE)
        assert ax.get_xlim()[0] <= x.min() and x.max() <= ax.get_xlim()[1]
        assert ax.get_ylim()[0] <= y.min() and y.max() <= ax.get_ylim()[1]
    assert data["regression"]["intercept"] == pytest.approx(.9512951279669726)
    assert data["regression"]["r2"] == pytest.approx(.8167176697920289)
    assert data["projection_regression"]["r2"] == pytest.approx(.946105)
    assert fig.axes[0].get_position().width / fig.axes[1].get_position().width == pytest.approx(55 / 45)
    assert fig.axes[0].get_ylabel() == "Full-model speedup (×)"
    assert len(fig.legends) == 1 and [t.get_text() for t in fig.legends[0].get_texts()] == list(kernels.STYLE)
    kernels.plt.close(fig)


def test_projection_gain_uses_raw_matched_latencies_and_pooled_instruction_counts():
    data = kernels.read_evidence()
    investigated = {p["condition"]: p for p in json.loads(kernels.INVESTIGATION.read_text())}
    points = data["projection_points"]
    assert [p["condition"] for p in points] == [f"c{i:02d}" for i in range(1, 31)]
    for p in points:
        record = investigated[p["condition"]]
        assert p["evidence_id"] == record["checkpoint_evidence_id"]
        bypassed = sum(record[f"{op}_mma_bypassed"] for op in ("qkv", "attn_out", "ffn_up", "ffn_down"))
        potential = sum(record[f"{op}_mma_potential"] for op in ("qkv", "attn_out", "ffn_up", "ffn_down"))
        assert p["bypass_percent"] == pytest.approx(100 * bypassed / potential)
        assert p["projection_sparse_gain"] == pytest.approx(record["all_skips_off_candidate_gm_ms"] / record["projection_on_candidate_gm_ms"])
    assert any(not np.isclose(p["projection_sparse_gain"], investigated[p["condition"]]["projection_sparse_gain_native_normalized"])
               for p in points)
    endpoint, = [p for p in points if p["condition"] == "c20"]
    assert endpoint["bypass_percent"] == pytest.approx(82.29771990388242)
    assert endpoint["projection_sparse_gain"] == pytest.approx(1.3782512496444377)
