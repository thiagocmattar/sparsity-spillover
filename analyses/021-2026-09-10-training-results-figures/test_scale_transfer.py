"""Check loss references, common-ceiling normalization and retained clipping data."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scales", HERE / "05_scale_transfer.py")
scales = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scales)


def test_approved_endpoints_and_high_threshold_ordering():
    data = scales.read_evidence()
    assert data == json.loads((HERE / "data/scale-transfer.json").read_text())
    assert scales.table_rows(data) == [
        ["14M", "27.48%", "91.8%", "+0.621"],
        ["70M", "40.60%", "82.2%", "+1.116"],
        ["410M", "80.62%", "92.4%", "+0.573"],
    ]
    assert data["coverage"] == dict(documents=500, sequences=338, input_tokens=692224,
                                    prediction_tokens=691886, excluded_tail_tokens=1444, seed_count=1)
    for panel in data["panels"]:
        four, seven = [r for r in panel["trained"] if r["dose"] == .5]
        assert seven["loss"] < four["loss"]
        assert seven["sparsity_percent"] > four["sparsity_percent"]
        assert all(scales.YLIM[0] <= r["delta_loss_vs_A0"] <= scales.YLIM[1]
                   for r in panel["trained"])


def test_all_recipes_use_common_A7_reach_and_unrounded_same_size_A0():
    for panel in scales.read_evidence()["panels"]:
        for row in panel["trained"] + panel["clipping"]:
            assert row["delta_loss_vs_A0"] == row["loss"] - panel["baseline"]["loss"]
            expected = row["sparsity_percent"] / panel["ceilings"]["A7"]["R_model_max_fraction"]
            assert row["A7_ceiling_used_percent"] == pytest.approx(expected)
        four = next(r for r in panel["trained"] if r["family"] == "A4-OL1" and r["dose"] == .5)
        own = four["sparsity_percent"] / panel["ceilings"]["A4"]["R_model_max_fraction"]
        assert four["A7_ceiling_used_percent"] < own


def test_clipping_zero_measurement_and_offscale_points_are_preserved():
    data = scales.read_evidence()
    # The separately evaluated 14M clipping p=0 differs slightly from untreated A0.
    zero = data["panels"][0]["clipping"][0]
    assert zero["delta_loss_vs_A0"] == pytest.approx(.000021047493410364382)
    assert zero["sparsity_percent"] > 0
    for panel in data["panels"]:
        assert [r["dose"] for r in panel["clipping"]] == [n / 10 for n in range(10)]
        assert panel["clipping_points_above_display"] == 4
    fig = scales.make_figure(data)
    for ax, panel in zip(fig.axes, data["panels"]):
        curve = next(line for line in ax.lines if line.get_gid() == f'{panel["scale"]}:clipping')
        assert len(curve.get_xdata()) == 10
        assert list(curve.get_ydata()) == [r["delta_loss_vs_A0"] for r in panel["clipping"]]
        assert ax.get_ylim() == scales.YLIM
    scales.plt.close(fig)
