"""Scientific checks for the fixed-reference appendix diagnostic."""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_appendix", HERE / "31_kernel_appendix.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_fixed_native_references_and_full_matched_membership():
    data = json.loads(analysis.DATA.read_text())
    old = json.loads(analysis.SPEED.read_text())
    assert len(data["points"]) == len(old["points"]) == 84
    expected = {(r["kind"], r["checkpoint_key"], r.get("target")) for r in old["points"]}
    actual = {("posthoc" if r["kind"] == "clipping" else "trained", r["checkpoint_key"], r["p"])
              for r in data["points"]}
    assert actual == expected
    for size, native in (("14M", .6554423980056099), ("70M", 1.6638832965631953)):
        rows = [r for r in data["points"] if r["model"] == size]
        assert len(rows) == 42
        assert {r["native_base_ms"] for r in rows} == {native}
        for r in rows:
            assert np.isclose(r["native_base_speedup"] * r["latency_ms"], native, rtol=1e-13)
        base, = [r for r in rows if r["kind"] == "trained" and r["scope"] == "0"]
        assert base["specialized_base_speedup"] == 1
        if size == "70M":
            assert .50 < base["native_base_speedup"] < .51
            assert sum(r["native_base_speedup"] > 1 for r in rows if r["kind"] == "trained") == 3


def test_count_pooling_retains_instruction_and_scalar_distinctions():
    data = json.loads(analysis.DATA.read_text())
    for r in data["points"]:
        for group, names in (("projection", analysis.PROJECTION), ("attention", analysis.ATTENTION)):
            raw = [r["operations"][n] for n in names]
            expected = sum(a["bypassed_mmas"] for a in raw) / sum(a["potential_mmas"] for a in raw)
            assert np.isclose(r[group]["bypass_percent"], 100 * expected, rtol=1e-14)
            assert r[group]["issued_mmas"] + r[group]["bypassed_mmas"] == r[group]["potential_mmas"]
        assert r["attention"]["simt_products"] == 0
    high, = [r for r in data["points"] if (r["model"], r["scope"], r["pressure"], r["kappa"]) == ("14M", "7", "all", .5)]
    assert high["projection"]["simt_products"] > 0
    assert not np.isclose(high["projection"]["bypass_percent"],
                          np.mean([high["operations"][n]["bypass_fraction"] * 100 for n in analysis.PROJECTION]))
    assert np.isclose(high["operations"]["qk_scores"]["bypass_fraction"], .5799510802063824, atol=1e-6)
    assert high["operations"]["probability_value"]["base_model_bypass_fraction"] > .10


def test_figure_contains_all_points_without_fits_and_no_clipping():
    data = json.loads(analysis.DATA.read_text())
    fig = analysis.make_figure(data)
    assert len(fig.axes) == 6
    for i, ax in enumerate(fig.axes):
        # Six trained recipe lines and two post-hoc paths precede the reference line.
        assert len(ax.lines) == (8 if i % 3 == 0 else 9)
        assert sum(len(line.get_xdata()) for line in ax.lines[:8]) == 42
        for line in ax.lines[:8]:
            assert ax.get_xlim()[0] <= min(line.get_xdata()) <= max(line.get_xdata()) <= ax.get_xlim()[1]
            assert ax.get_ylim()[0] <= min(line.get_ydata()) <= max(line.get_ydata()) <= ax.get_ylim()[1]
    assert data["script_sha256"] == analysis.sha(HERE / "31_kernel_appendix.py")
    assert data["output_sha256"] == analysis.sha(analysis.OUTPUT)
    for relative, digest in data["sources_sha256"].items():
        assert analysis.sha(analysis.ROOT / relative) == digest
    analysis.plt.close(fig)
