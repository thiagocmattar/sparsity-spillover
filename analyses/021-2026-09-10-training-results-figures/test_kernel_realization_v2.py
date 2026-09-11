"""Verify that v2 changes the predictor, not the cohort or runtime ratio."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernels_v2", HERE / "07_kernel_realization_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def test_smodel_and_projection_gains_match_the_same_original_checkpoints():
    data = v2.read_evidence()
    assert data == json.loads((HERE / "data/kernel-realization-v2.json").read_text(encoding="utf-8"))
    original = json.loads((HERE / "data/kernel-realization.json").read_text(encoding="utf-8"))
    assert data["points"] == original["points"]
    assert data["regression"] == original["regression"]
    assert data["original_figure_sha256"] == hashlib.sha256((HERE / "figures/07-kernel-realization.pdf").read_bytes()).hexdigest()
    full = {p["condition"]: p for p in data["points"] if p["candidate"] == "k050"}
    assert len(data["projection_points"]) == 30
    for p, previous in zip(data["projection_points"], original["projection_points"]):
        assert {k: value for k, value in p.items() if k != "sparsity_percent"} == previous
        counts = full[p["condition"]]["canonical_counts"]
        assert p["sparsity_percent"] == pytest.approx(100 * counts["block_zero_product_count"] / counts["model_product_count"])
        assert p["projection_sparse_gain"] == pytest.approx(p["all_skips_off_candidate_gm_ms"] / p["projection_on_candidate_gm_ms"])


def test_both_panels_use_smodel_and_projection_fit_is_recomputed():
    data = v2.read_evidence()
    points = data["projection_points"]
    x = np.array([p["sparsity_percent"] for p in points])
    y = np.array([p["projection_sparse_gain"] for p in points])
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(30), x]), y, rcond=None)[0]
    fit = data["projection_regression"]
    assert [fit["intercept"], fit["slope_per_percentage_point"]] == pytest.approx(coefficients)
    assert fit["r2"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
    assert fit["r2"] == pytest.approx(.4957330004096926)
    fig = v2.base.make_figure(data, projection_x=data["projection_x_metric"])
    assert fig.axes[0].get_xlabel() == fig.axes[1].get_xlabel()
    for ax in fig.axes:
        plotted = np.concatenate([collection.get_offsets() for collection in ax.collections])
        assert sorted(plotted[:, 0]) == pytest.approx(sorted(x))
    plotted = np.concatenate([collection.get_offsets() for collection in fig.axes[1].collections])
    assert sorted(map(tuple, plotted)) == sorted(zip(x, y))
    line, = [line for line in fig.axes[1].lines if line.get_gid() == "OLS"]
    assert line.get_ydata() == pytest.approx(coefficients[0] + coefficients[1] * line.get_xdata())
    v2.base.plt.close(fig)
