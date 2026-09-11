"""Check complete A0 histories, the pre-clip estimand, and smoothing boundaries."""

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("a0", HERE / "06_a0_optimization.py")
a0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a0)


def test_complete_step_and_token_coverage_from_source_logs():
    data = a0.read_evidence()
    assert data == json.loads((HERE / "data/a0-optimization.json").read_text())
    assert sum(len(r["rows"]) for r in data["runs"]) == 2136
    for run in data["runs"]:
        assert [e["step"] for e in run["rows"]] == list(range(1, 713))
        np.testing.assert_array_equal(run["training_tokens_billions"], np.arange(1, 713) * 2097152 / 1e9)
        assert run["rows"][-1]["input_tokens_seen"] == 1493172224
        assert not any(e["optimizer_step_skipped"] or e["gradient_overflow"] for e in run["rows"])
        assert set(run["identity"]["seeds"].values()) == {1234}


def test_pre_clipping_norms_and_training_losses_are_preserved_in_plots():
    data = a0.read_evidence()
    np.testing.assert_allclose([r["summary"]["final_training_loss"] for r in data["runs"]],
                               [5.243930354714394, 3.9830125523731112, 4.448348488658667])
    np.testing.assert_allclose([r["summary"]["maximum_pre_clip_norm"] for r in data["runs"]],
                               [2.001891851425171, 3.513155698776245, 26.9615478515625])
    assert [r["summary"]["clipped_steps"] for r in data["runs"]] == [5, 8, 56]
    fig = a0.make_figure(data)
    for ax, field in zip(fig.axes, ["task_loss", "adamw_gradient_norm_pre_clip"]):
        for run in data["runs"]:
            line = next(l for l in ax.lines if l.get_gid() == f'{run["scale"]}:{field}:raw')
            np.testing.assert_array_equal(line.get_ydata(), [e[field] for e in run["rows"]])
        assert ax.get_xlim() == (0, 1.5)
    assert fig.axes[1].get_yscale() == "log"
    a0.plt.close(fig)


def test_centered_smoothing_uses_partial_edges_without_zero_padding():
    np.testing.assert_allclose(a0.smooth([1, 2, 3, 4, 5], window=3), [1.5, 2, 3, 4, 4.5])
    np.testing.assert_array_equal(a0.smooth(np.ones(20)), np.ones(20))
    values = np.arange(712, dtype=float)
    smoothed = a0.smooth(values)
    assert smoothed[100] == np.mean(values[96:105])
    assert smoothed[0] == np.mean(values[:5])
    assert smoothed[-1] == np.mean(values[-5:])
