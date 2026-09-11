"""Verify retained evidence, timing estimands and the diagnostic distinctions."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load_script("01_extract")
analyzer = load_script("02_analyze")


@pytest.fixture(scope="module")
def evidence():
    return extractor.extract()


def test_retained_sources_and_complete_export(evidence):
    rows, layers, provenance = evidence
    assert rows == json.loads((HERE / "data/checkpoints.json").read_text())
    assert provenance == json.loads((HERE / "data/provenance.json").read_text())
    assert len(layers) == 180
    assert [r["condition"] for r in rows] == [f"c{i:02d}" for i in range(1, 31)]
    assert {f: sum(r["family"] == f for r in rows) for f in analyzer.FAMILIES} == dict.fromkeys(analyzer.FAMILIES, 10)
    assert len([p for p in provenance["sources"] if p.endswith("/timing.json")]) == 270
    assert len([p for p in provenance["sources"] if p.endswith("/diagnostics.json")]) == 30
    with (HERE / "data/checkpoints.csv").open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 30
    for row, record in zip(rows, exported):
        for field, value in row.items():
            assert record[field] == ("" if value is None else str(value))
        for field in ("projection_pure_zero_only_bypass_fraction", "projection_timing_subset_bypass_fraction",
                      "attention_unmasked_only_mma_skip_fraction", "attention_timing_subset_mma_skip_fraction"):
            assert row[field] is None and record[field] == ""


def test_geometric_mean_timing_and_pair_coverage():
    samples, native, candidate = [], [], []
    for i in range(448):
        n, c = (1 if i < 300 else 9), (1 if i < 148 else 2)
        native.append(n); candidate.append(c)
        for mode, value in (("native_graph", n), ("candidate_graph", c)):
            samples.append(dict(mode=mode, repeat=i // 64, input_index=i % 64,
                                host_ms=value, output_shape=[1, 2048, 50304]))
    timing = dict(indices=list(range(64)), samples=samples)
    result = extractor.timing_summary(timing)
    expected = np.exp(np.mean(np.log(np.array(native) / candidate)))
    assert result["paired_speedup"] == pytest.approx(expected)
    assert result["native_gm_ms"] / result["candidate_gm_ms"] == pytest.approx(expected)
    assert not np.isclose(expected, np.median(native) / np.median(candidate))
    with pytest.raises(AssertionError):
        extractor.timing_summary({**timing, "samples": samples + samples[:1]})
    with pytest.raises(AssertionError):
        extractor.timing_summary({**timing, "samples": samples[:-1]})


def test_counts_use_distinct_scalar_and_instruction_denominators(evidence):
    rows, _, _ = evidence
    for r in rows:
        assert sum(r[f"{op}_model_contribution_pp"] for op in extractor.OPS) == pytest.approx(r["s_model_percent"])
        assert r["projection_model_contribution_pp"] + r["attention_model_contribution_pp"] == pytest.approx(r["s_model_percent"])
        assert r["projection_mma_bypassed"] == sum(r[f"{op}_mma_bypassed"] for op in extractor.PROJECTIONS)
        for op in extractor.PROJECTIONS:
            products_per_mma = 1024 if op in ("ffn_down", "attn_out") else 2048
            assert r[f"{op}_mma_potential"] * products_per_mma == r[f"{op}_products"]
            assert r[f"{op}_mma_potential"] == r[f"{op}_mma_issued"] + r[f"{op}_mma_bypassed"]
            assert 0 <= r[f"{op}_zero_row_fraction"] <= r[f"{op}_row_le2_fraction"] <= 1
        assert r["qkv_zero_16x16_fragments"] * 48 == r["qkv_mma_bypassed"]
        assert r["ffn_up_zero_16x16_fragments"] * 64 == r["ffn_up_mma_bypassed"]
        assert r["ffn_down_simt_safety_gap_products"] == r["attn_out_simt_safety_gap_products"] == 0
        assert r["fusion_only_speedup_common_native"] * r["projection_sparse_gain"] * r["attention_sparse_gain"] == pytest.approx(r["full_speedup"])
        assert r["attention_sparse_gain"] < 1
    assert all(r["qk_mma_skip_fraction"] == 0 and r["pv_mma_skip_fraction"] == pytest.approx(5 / 48) for r in rows[:-1])
    assert rows[-1]["qk_mma_skip_fraction"] == pytest.approx(.5799516539497384)
    assert rows[-1]["pv_mma_skip_fraction"] == pytest.approx(.6720844396707809)


def test_matched_comparisons_separate_speedup_from_absolute_latency(evidence):
    rows, _, _ = evidence
    associations, pairs, summary = analyzer.analyze(rows)
    assert dict(associations=associations, matched_pairs=pairs, family_summary=summary) == json.loads((HERE / "data/analysis.json").read_text())
    assert len(pairs) == 10
    for p in pairs:
        assert p["seven_over_four_full_speedup"] == pytest.approx(p["seven_over_four_native_baseline_gm_ms"] / p["seven_over_four_full_candidate_gm_ms"])
        if p["kappa"] > 0:
            assert p["seven_over_four_full_speedup"] > 1
            assert p["seven_over_four_full_candidate_gm_ms"] > 1
        if p["pressure"] == "orthogonal_l1":
            assert p["delta_projection_model_contribution_pp"] < 0
            assert p["delta_projection_mma_bypass_fraction"] < 0
            assert p["delta_projection_sparse_gain"] < 0
    four, seven = rows[19], rows[29]
    assert (four["condition"], seven["condition"]) == ("c20", "c30")
    assert four["full_speedup"] == pytest.approx(1.609320804244549)
    assert seven["full_speedup"] == pytest.approx(1.7831745639788332)
    assert four["full_candidate_gm_ms"] < seven["full_candidate_gm_ms"]


def test_correlations_do_not_assign_r_squared_to_constant_skip_counts(evidence):
    rows, _, _ = evidence
    correlations, _, _ = analyzer.analyze(rows)
    for result in correlations:
        if result["ols_r2"] is not None:
            assert result["ols_r2"] == pytest.approx(result["pearson_r"] ** 2)
        if result["predictor"] == "attention_mma_skip_fraction":
            if result["family"] in ("baseline/local", "4-site"):
                assert result["ols_r2"] is None and result["predictor_unique_values"] == 1
            else:
                assert result["predictor_unique_values"] == 2
    full = next(r for r in correlations if r["predictor"] == "s_model_percent" and r["outcome"] == "full_speedup" and r["family"] == "all")
    assert full["ols_r2"] == pytest.approx(.8167176697920289)
