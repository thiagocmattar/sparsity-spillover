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


def test_figure_retains_each_logged_pair_and_unsmoothed_traces(evidence):
    conditions, provenance = evidence
    fig = geometry.make_figure(conditions, geometry.summary_document(conditions, provenance))
    left, right = fig.axes
    rows = [r for c in conditions for r in c["rows"]]
    expected = np.array([[r["task_pressure_cosine_before"], r["task_pressure_cosine_after"]]
                         for r in rows])
    np.testing.assert_array_equal(left.collections[0].get_offsets(), expected)
    assert np.any(expected[:, 1] != 0)  # Keep measured residuals, not imposed orthogonality.
    ratios = np.array([[r["pressure_to_task_ratio_raw"] / c["pressure"]["step_budget"]
                        for r in c["rows"]] for c in conditions])
    for line, values in zip(right.lines[:10], ratios):
        np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 713))
        np.testing.assert_array_equal(line.get_ydata(), values)
    np.testing.assert_array_equal(right.lines[10].get_ydata(), np.median(ratios, axis=0))
    assert right.get_yscale() == "log"
    geometry.plt.close(fig)
