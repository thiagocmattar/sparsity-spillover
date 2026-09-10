"""Check cohort selection, count accounting, and plotted source coordinates."""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("overview", HERE / "01_quality_sparsity.py")
overview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overview)


@pytest.fixture(scope="module")
def evidence():
    return json.loads((HERE / "data/14m-quality-sparsity.json").read_text(encoding="utf-8"))


def test_export_matches_sources_and_requested_selection(evidence):
    assert evidence == overview.read_evidence()
    trained = [row for row in evidence["points"] if row["kind"] == "trained"]
    clipped = [row for row in evidence["points"] if row["kind"] == "clipped"]
    assert len(trained) == 26
    assert len(clipped) == 20
    assert len({row["id"] for row in evidence["points"]}) == 46
    assert {row["family"] for row in trained} == {"A0", "A1-H", "A1-H-OL1", "A4", "A4-OL1", "A7", "A7-OL1"}
    for family in ("A0", "A1-H"):
        assert [row["dose"] for row in overview.series(evidence, family, "clipped")] == [i / 10 for i in range(10)]
        assert sum(row["loss"] > overview.YLIM[1] for row in clipped if row["family"] == family) == 4
        # The measured p=0 evaluation must not be snapped to its training endpoint.
        assert overview.series(evidence, family, "clipped")[0]["loss"] != overview.series(evidence, family)[0]["loss"]
    for row in evidence["points"]:
        assert row["coverage"]["sequences"] == 338
        assert row["coverage"]["input_tokens"] == 338 * 2048
        assert row["coverage"]["excluded_tail_tokens"] == 1444
        assert row["counts"]["model_product_count"] == 6363055915008
        assert row["R_model"] == row["counts"]["block_zero_product_count"] / row["counts"]["model_product_count"]
        if row["kind"] == "clipped":
            assert row["clipping_sites"] == ["a", "m", "h", "z"]


def test_reach_guides_include_projections_attention_and_output_denominator(evidence):
    # Independent Pythia-14M scalar-product calculation, including causal masking.
    tokens, width, ffn, layers, vocabulary = 2048, 128, 512, 6, 50304
    projections = layers * tokens * (4 * width**2 + 2 * width * ffn)
    attention = layers * width * tokens * (tokens + 1)
    denominator = projections + attention + tokens * width * vocabulary
    for row, reachable in zip(evidence["ceilings"], (projections, projections + attention)):
        assert row["reachable_product_count"] == reachable
        assert row["model_product_count"] == denominator
        assert row["R_model_max_fraction"] == reachable / denominator
    assert evidence["ceilings"][0]["active_sites"] == ["a", "m", "h", "z"]
    assert evidence["ceilings"][1]["active_sites"] == ["a", "m", "h", "q_post", "k_post", "v", "z"]


def test_artist_coordinates_preserve_every_evaluated_setting(evidence):
    fig, ax = overview.make_figure(evidence)
    artists = {line.get_gid(): line for line in ax.lines}
    try:
        for family in overview.FAMILIES:
            expected = overview.series(evidence, family)
            np.testing.assert_array_equal(artists[f"trained:{family}"].get_xydata(), [overview.xy(row) for row in expected])
        for family in ("A0", "A1-H"):
            expected = overview.series(evidence, family, "clipped")
            np.testing.assert_array_equal(artists[f"clipped:{family}"].get_xydata(), [overview.xy(row) for row in expected])
        for row in evidence["ceilings"]:
            np.testing.assert_array_equal(artists[f"ceiling:{row['family']}"].get_xdata(), [row["R_model_max_percent"]] * 2)
    finally:
        plt.close(fig)
