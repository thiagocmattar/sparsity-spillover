"""Check the review's changed estimands against retained measurements."""
import importlib.util
import json
from pathlib import Path

import matplotlib.colors as colors
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("review_figures", HERE / "10_review_figures.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


@pytest.fixture(scope="module")
def evidence():
    return review.read_evidence()


def test_separate_sites_preserve_counts_and_density_mass(evidence):
    saved = json.loads((HERE / "data/review-corrections.json").read_text(encoding="utf-8"))
    for key, value in evidence.items():
        assert saved[key] == value
    for row in evidence["density"]:
        groups = row["groups"]
        assert groups["h"]["total"] == 4 * groups["m"]["total"]
        assert groups["qkv"]["total"] == 3 * groups["m"]["total"]
        for group in groups.values():
            assert sum(group["bin_counts"]) + group["exact_zero_count"] + group["underflow"] + group["overflow"] == group["total"]
        for key, sites in (("m", {"m"}), ("h", {"h"}), ("qkv", {"q_post", "k_post", "v"})):
            cells = [x for x in row["layers"] if x["name"].split(".")[0] in sites]
            assert sum(x["total"] for x in cells) == groups[key]["total"]
            assert sum(x["exact_zero_count"] for x in cells) == groups[key]["exact_zero_count"]
    four, seven = evidence["density"][1:]
    assert four["groups"]["m"]["zero_percent"] - seven["groups"]["m"]["zero_percent"] > 28
    assert abs(four["groups"]["h"]["zero_percent"] - seven["groups"]["h"]["zero_percent"]) < .07
    fig = review.densities(evidence)
    fig.canvas.draw()
    for ax in fig.axes[:3]:
        assert len(ax.patches) == 3
        for text in ax.texts:
            assert fig.bbox.contains(*text.get_window_extent().get_points()[0])
            assert fig.bbox.contains(*text.get_window_extent().get_points()[1])
    review.plt.close(fig)


def test_pressure_free_controls_interaction_and_quality_cost(evidence):
    source = json.loads((review.ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json").read_text(encoding="utf-8"))
    points = {r["id"]: r for r in source["trained"]}
    assert [r["family"] for r in evidence["operations"]] == ["A4", "A4-OL1", "A7", "A7-OL1"]
    for row in evidence["operations"]:
        raw = points[row["source_id"]]["counts"]
        assert sum(row["contributions_pp"].values()) == pytest.approx(100 * sum(r["zero_product_count"] for r in raw["per_operation"].values()) / raw["model_product_count"])
    interaction = evidence["interaction_at_kappa_0_5"]
    assert interaction["sparsity_pp"] == pytest.approx(9.597959609179894)
    assert interaction["loss"] == pytest.approx(-.2517925289255629)
    for row, expected in zip(evidence["scale_quality_costs"], [.6208335686717517, 1.1161586970267212, .5732352310383817]):
        assert row["delta_loss"] == pytest.approx(expected)
        assert row["perplexity_ratio"] == pytest.approx(np.exp(expected))
    # Directly verify the simplified seven-site formula against the full counts.
    for layers, width in ((6, 128), (6, 512), (24, 1024)):
        t, vocab = 2048, 50304
        block = 12 * t * width**2 + width * t * (t + 1)
        reach = layers * block / (layers * block + t * width * vocab)
        assert reach == pytest.approx(layers * (12 * width + t + 1) / (layers * (12 * width + t + 1) + vocab))


def test_kernel_uses_absolute_projection_only_latency_and_correct_fill(evidence):
    rows = evidence["runtime"]
    fig = review.kernel(evidence)
    a, b = fig.axes
    plotted = np.concatenate([c.get_offsets() for c in a.collections[:3]])
    np.testing.assert_allclose(sorted(map(tuple, plotted)), sorted((p["projection_on_candidate_gm_ms"], p["validation_loss"]) for p in rows))
    for collection, family, color in zip(a.collections[:3], ("baseline/local", "4-site", "7-site"), (review.GRAY, review.BLUE, review.ORANGE)):
        expected = [colors.to_rgba(color if family == "baseline/local" or p["pressure"] == "orthogonal_l1" else "white") for p in rows if p["family"] == family]
        np.testing.assert_allclose(collection.get_facecolors(), expected)
    for p in rows:
        assert p["projection_sparse_gain"] == pytest.approx(p["all_skips_off_candidate_gm_ms"] / p["projection_on_candidate_gm_ms"])
        assert p["projection_on_candidate_gm_ms"] < p["full_candidate_gm_ms"]
    x = np.array([p["projection_mma_bypass_fraction"] for p in rows])
    y = np.array([p["projection_sparse_gain"] for p in rows])
    assert np.corrcoef(x, y)[0, 1] ** 2 == pytest.approx(.946, abs=.0005)
    assert max(rows, key=lambda p: p["s_model_percent"])["condition"] == "c30"
    assert min(rows, key=lambda p: p["projection_on_candidate_gm_ms"])["condition"] == "c20"
    review.plt.close(fig)
