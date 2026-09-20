"""Scientific coverage and source reconciliation for the OL1 appendix."""

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ol1_appendix", HERE / "29_ol1_appendix.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_export_matches_every_figure1_boundary_and_preserves_zero_diagnostics():
    sources = {}
    checkpoints = analysis.read_json(analysis.COHORT, sources)["checkpoints"]
    figure1 = analysis.read_json(analysis.FIGURE1, sources)
    conditions, families, verified = analysis.read_geometry(checkpoints, figure1, sources)
    saved = json.loads(analysis.DATA.read_text())
    assert conditions == saved["geometry"]["conditions"]
    assert families == saved["geometry"]["families"]
    assert verified == saved["verified_historical_code"]
    assert sum(c["observations"] for c in conditions) == 28480
    for condition in conditions:
        trace = condition["trace"]
        assert len(trace["rho_opp"]) == len(trace["r_over_b"]) == len(trace["trust_scale"]) == 712
        assert sum(s < 1 for s in trace["trust_scale"]) == condition["cap_active_steps"]
        assert np.median(trace["rho_opp"]) == condition["rho_opp"]["median"]
    zero_median = next(c for c in conditions if
                      (c["model"], c["scope"], c["pressure"], c["kappa"]) == ("70M", "4", "all", .01))
    assert zero_median["rho_opp"]["median"] == 0
    assert sum(v == 0 for v in zero_median["trace"]["rho_opp"]) >= 356
    assert families["70M/T7/Pall"]["cap_active_steps"] == 3560
    assert families["14M/T7/Pall"]["cap_active_steps"] == 3453
    assert all(s["zero_learning_rate_steps"] == 5 for s in families.values())


def test_table_uses_ordinary_final_loss_and_identical_training_protocol():
    sources = {}
    checkpoints = analysis.read_json(analysis.COHORT, sources)["checkpoints"]
    comparison = analysis.read_comparison(checkpoints, sources)
    saved = json.loads(analysis.DATA.read_text())
    assert comparison == saved["t1_comparison"]
    differences = []
    for weight in (.05, .1, .5, 1.):
        a, b = [next(r for r in comparison["rows"] if r["lambda"] == weight and r["method"] == method)
                for method in ("L1", "OL1")]
        assert abs(a["sparsity_percent"] - b["sparsity_percent"]) < .02
        differences.append(b["validation_loss"] - a["validation_loss"])
    assert all(d < 0 for d in differences[:3]) and differences[3] > 0
    for row in comparison["rows"]:
        assert row["loss_evaluation"]["pass"] == "ordinary reloaded final checkpoint"


def test_figure_has_only_14m_and_preserves_every_recipe_quantile(monkeypatch, tmp_path):
    saved = json.loads(analysis.DATA.read_text())
    conditions = saved["geometry"]["conditions"]
    assert saved["geometry"]["plotted_models"] == ["14M"]
    assert len(saved["geometry"]["plotted_checkpoint_keys"]) == 20
    monkeypatch.setattr(analysis, "OUTPUT", tmp_path / "geometry.pdf")
    figure = analysis.make_figure(conditions)
    assert len(figure.axes) == 2
    left, right = figure.axes
    assert len(left.containers) == 4 and len(right.lines) == 25
    for i, (scope, pressure, _, color, _) in enumerate(analysis.STYLES):
        group = sorted((r for r in conditions if (r["model"], r["scope"], r["pressure"]) ==
                        ("14M", scope, pressure)), key=lambda r: r["kappa"])
        medians = 100 * np.array([np.median(r["trace"]["rho_opp"]) for r in group])
        dots, _, bars = left.containers[i].lines
        np.testing.assert_array_equal(dots.get_ydata(), medians)
        intervals = np.asarray(bars[0].get_segments())[:, :, 1]
        assert np.min(intervals) > left.get_ylim()[0]
        assert np.max(intervals) < left.get_ylim()[1]
        ratios = np.array([r["trace"]["r_over_b"] for r in group])
        np.testing.assert_array_equal(right.lines[6*i+5].get_ydata(), np.median(ratios, axis=0))
        assert ratios.min() > right.get_ylim()[0] and ratios.max() < right.get_ylim()[1]
        assert dots.get_color() == right.lines[6*i+5].get_color() == color
    analysis.plt.close(figure)


def test_captioned_outputs_and_provenance_match_files():
    saved = json.loads(analysis.DATA.read_text())
    assert saved["script_sha256"] == analysis.sha(HERE / "29_ol1_appendix.py")
    for path, digest in saved["source_sha256"].items():
        assert analysis.sha(analysis.ROOT / path) == digest
    for path, digest in saved["output_sha256"].items():
        assert analysis.sha(analysis.ROOT / path) == digest
    for relative in ("figures/18-14m-70m-ol1-geometry.pdf", "tables/t1-l1-ol1.tex"):
        manuscript = analysis.ROOT / "manuscript/draft" / relative
        source = analysis.OUTPUT if manuscript.suffix == ".pdf" else analysis.TABLE
        assert manuscript.read_bytes() == source.read_bytes()
