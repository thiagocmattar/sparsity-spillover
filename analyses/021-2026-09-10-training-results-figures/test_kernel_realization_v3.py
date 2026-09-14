"""Common-reference normalization must not change the matched ablation panel."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernels_v3", HERE / "07_kernel_realization_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def test_one_native_a0_reference_and_unchanged_checkpoint_latencies():
    data = v3.read_evidence()
    assert data == json.loads((HERE / "data/kernel-realization-v3.json").read_text(encoding="utf-8"))
    rows = json.loads(v3.base.INVESTIGATION.read_text(encoding="utf-8"))
    source = {p["condition"]: p for p in rows}
    reference = source["c01"]["full_native_gm_ms"]
    points = data["common_a0_points"]
    assert {p["condition"] for p in points} == {f"c{i:02d}" for i in range(1, 31)}
    assert {p["common_a0_native_gm_ms"] for p in points} == {reference}
    for point in points:
        original = source[point["condition"]]
        assert point["full_candidate_gm_ms"] == original["full_candidate_gm_ms"]
        assert point["a0_relative_speedup"] == pytest.approx(reference / original["full_candidate_gm_ms"])
        assert point["sparsity_percent"] == pytest.approx(original["s_model_percent"])
    # Equal numerators must rank speedups by inverse absolute optimized latency.
    assert [p["condition"] for p in sorted(points, key=lambda p: -p["a0_relative_speedup"])] == [
        p["condition"] for p in sorted(points, key=lambda p: p["full_candidate_gm_ms"])]


def test_refitted_panel_a_and_identical_panel_b():
    data = v3.read_evidence()
    previous = v3.v2.read_evidence()
    for key in ("points", "regression", "projection_points", "projection_regression",
                "projection_sweeps", "local_projection_sweeps"):
        assert data[key] == previous[key]
    points = data["common_a0_points"]
    x = np.array([p["sparsity_percent"] for p in points])
    y = np.array([p["a0_relative_speedup"] for p in points])
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(30), x]), y, rcond=None)[0]
    fit = data["common_a0_regression"]
    assert [fit["intercept"], fit["slope_per_percentage_point"]] == pytest.approx(coefficients)
    assert fit["r2"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
    assert fit["r2"] == pytest.approx(.5099165323360964)
    fig = v3.make_figure(data)
    before = v3.v2.make_figure(previous)
    plotted = np.concatenate([c.get_offsets() for c in fig.axes[0].collections])
    np.testing.assert_allclose(sorted(map(tuple, plotted)), sorted(zip(x, y)))
    for current, original in zip(fig.axes[1].collections, before.axes[1].collections, strict=True):
        np.testing.assert_array_equal(current.get_offsets(), original.get_offsets())
        np.testing.assert_array_equal(current.get_facecolors(), original.get_facecolors())
    for current, original in zip(fig.axes[1].lines, before.axes[1].lines, strict=True):
        np.testing.assert_array_equal(current.get_xydata(), original.get_xydata())
        assert current.get_linestyle() == original.get_linestyle()
    assert [t.get_text() for t in fig.axes[1].texts] == [t.get_text() for t in before.axes[1].texts]
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "1× native A0" in labels
    assert any("1.38" in label and "7-site" in label for label in labels)
    assert not any("1.78" in label or "0.817" in label for label in labels)
    v3.base.plt.close(fig)
    v3.base.plt.close(before)
