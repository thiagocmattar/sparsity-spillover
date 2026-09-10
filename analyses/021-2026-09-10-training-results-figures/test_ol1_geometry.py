"""Check retained boundary coverage, cap semantics, and plotted observations."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ol1_geometry", HERE / "02_ol1_geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


@pytest.fixture(scope="module")
def evidence():
    return geometry.read_evidence()


def test_complete_cohort_and_pooled_counts(evidence):
    conditions, provenance = evidence
    summary = geometry.summary_document(conditions, provenance)
    assert summary == json.loads((HERE / "data/14m-ol1-geometry.json").read_text(encoding="utf-8"))
    assert len(provenance["source_sha256"]) == 21
    assert len(provenance["verified_historical_code"]) == 8
    assert summary["pooled"]["observations"] == 7120
    assert summary["pooled"]["conflict_steps"] == 7111
    assert summary["pooled"]["cap_active_steps"] == 3471
    assert summary["by_family"]["A4-OL1"]["cap_active_steps"] == 18
    assert summary["by_family"]["A7-OL1"]["cap_active_steps"] == 3453
    assert summary["by_family"]["A4-OL1"]["median_pre_cosine"] == pytest.approx(-.034589598296058524)
    assert summary["by_family"]["A7-OL1"]["median_pre_cosine"] == pytest.approx(-.011574692682206221)
    assert summary["pooled"]["p99_abs_conflicting_post_cosine"] == pytest.approx(7.387589453976291e-9)
    assert summary["by_family"]["A4-OL1"]["rho_opp"]["median"] == pytest.approx(.00950305897111545)
    assert summary["by_family"]["A7-OL1"]["rho_opp"]["median"] == pytest.approx(.5880055965343591)
    assert summary["by_family"]["A4-OL1"]["rho_opp"]["above_one_steps"] == 0
    assert summary["by_family"]["A7-OL1"]["rho_opp"]["above_one_steps"] == 753
    rows = [row for condition in conditions for row in condition["rows"]]
    aligned = [r for r in rows if r["task_pressure_dot_before"] >= 0]
    assert len(aligned) == 9
    assert all(r["task_pressure_cosine_before"] == r["task_pressure_cosine_after"] for r in aligned)
    assert summary["pooled"]["max_abs_conflicting_post_cosine"] < 1.5e-8


def test_cap_count_uses_logged_scale_at_stabilized_boundary(evidence):
    # r < b can still bind because the implementation adds epsilon to r.
    row = copy.deepcopy(evidence[0][0]["rows"][1])
    row["task_pressure_dot_before"] = -1
    row["pressure_to_task_ratio_raw"] = 1 - .5e-12
    row["trust_scale"] = 1 / (row["pressure_to_task_ratio_raw"] + 1e-12)
    assert row["pressure_to_task_ratio_raw"] < 1
    assert geometry.summarize([row])["cap_active_steps"] == 1


def test_audit_rejects_wrong_geometry_or_skipped_boundary(evidence):
    condition = evidence[0][0]
    row = condition["rows"][1]
    for field, value in (("task_pressure_cosine_after", .1),
                         ("projection_applied", not row["projection_applied"]),
                         ("optimizer_step_skipped", True), ("trust_scale", .01)):
        corrupt = dict(row, **{field: value})
        with pytest.raises(AssertionError):
            geometry.audit_step(corrupt, condition["pressure"])


def test_opposing_component_matches_removed_vector_and_handles_sign():
    u = np.array([3.0, 4.0])
    w = np.array([-2.0, -11.0])
    dot = float(u @ w)
    row = {"task_pressure_dot_before": dot, "task_direction_norm": np.linalg.norm(u)}
    eps = 1e-3  # Make the stabilizer's placement numerically observable.
    rho = geometry.opposing_component_ratio(row, eps)
    projected = w - dot / (u @ u + eps) * u
    assert rho == pytest.approx(50 / (25 + eps), rel=1e-14)
    assert np.linalg.norm(projected - w) / np.linalg.norm(u) == pytest.approx(rho, rel=1e-14)
    for nonnegative_dot in (0.0, 50.0):
        assert geometry.opposing_component_ratio(dict(row, task_pressure_dot_before=nonnegative_dot), eps) == 0


def test_figure_retains_per_threshold_quantiles_and_separate_family_traces(evidence):
    conditions, provenance = evidence
    fig = geometry.make_figure(conditions, geometry.summary_document(conditions, provenance))
    left, right = fig.axes
    assert len(left.containers) == 2  # Two sets of five medians and empirical IQRs.
    assert len(right.lines) == 13  # Ten runs, two family medians, one budget reference.
    for i, family in enumerate(geometry.FAMILIES):
        cohort = [c for c in conditions if c["family"] == family]
        cosine_quantiles = np.array([np.quantile([r["task_pressure_cosine_before"] for r in c["rows"]],
                                                [.25, .5, .75]) for c in cohort])
        dots, caps, bars = left.containers[i].lines
        positions = np.arange(5) + (-.055 if i == 0 else .055)
        np.testing.assert_array_equal(dots.get_xdata(), positions)
        np.testing.assert_array_equal(dots.get_ydata(), cosine_quantiles[:, 1])
        segments = np.asarray(bars[0].get_segments())
        np.testing.assert_array_equal(segments[:, 0, 0], positions)
        np.testing.assert_allclose(segments[:, :, 1], cosine_quantiles[:, [0, 2]], rtol=1e-14)
        ratios = np.array([[r["pressure_to_task_ratio_raw"] / c["pressure"]["step_budget"]
                            for r in c["rows"]] for c in cohort])
        lines = right.lines[6 * i:6 * i + 6]
        for line, values in zip(lines[:5], ratios):
            np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 713))
            np.testing.assert_array_equal(line.get_ydata(), values)
        np.testing.assert_array_equal(lines[5].get_ydata(), np.median(ratios, axis=0))
        assert all(line.get_color() == dots.get_color() == geometry.COLORS[family] for line in lines)
    assert right.get_yscale() == "log"
    geometry.plt.close(fig)
