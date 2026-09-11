"""Check matched contrasts, count-derived units, and protection against mispairing."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paired", HERE / "03_paired_interventions.py")
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def test_source_pairs_reproduce_approved_values_without_rounding_coordinates():
    data = paired.read_evidence()
    assert data == json.loads((HERE / "data/14m-paired-interventions.json").read_text())
    assert len(data["endpoints"]) == 20
    assert data["coverage"] == dict(documents=500, sequences=338, input_tokens=692224,
                                    prediction_tokens=691886, excluded_tail_tokens=1444, seed_count=1)
    expected = {
        "A4": ([-.012, -.008, .056, .129, .378], [.60, 1.17, 2.25, 2.44, 2.50]),
        "A7": ([.012, .017, .025, .001, .127], [-.16, .11, .74, 1.37, 12.10]),
    }
    for family, (loss, sparsity) in expected.items():
        rows = [r for r in data["pairs"] if r["family"] == family]
        assert [r["kappa"] for r in rows] == list(paired.KAPPAS)
        np.testing.assert_allclose([r["delta_loss"] for r in rows], loss, atol=.0005, rtol=0)
        np.testing.assert_allclose([r["delta_sparsity_pp"] for r in rows], sparsity, atol=.005, rtol=0)


def test_pairs_are_keyed_by_family_and_threshold_not_source_order():
    rows = json.loads(paired.SOURCE.read_text())["trained"]
    assert paired.paired_effects(list(reversed(rows)))[1] == paired.paired_effects(rows)[1]


def test_unmatched_initialization_and_missing_threshold_are_rejected():
    rows = json.loads(paired.SOURCE.read_text())["trained"]
    target = next(r for r in rows if r["scale"] == "14M" and r["family"] == "A4-OL1")
    bad = copy.deepcopy(rows)
    bad[rows.index(target)]["identity"]["initial_parameter_sha256"] = "different-initialization"
    with pytest.raises(AssertionError, match="Unmatched initial_parameter_sha256"):
        paired.paired_effects(bad)
    with pytest.raises(AssertionError, match="Incomplete or duplicate"):
        paired.paired_effects([r for r in rows if r is not target])
