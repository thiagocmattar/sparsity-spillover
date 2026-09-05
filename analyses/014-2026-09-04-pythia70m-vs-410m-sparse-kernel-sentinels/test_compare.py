from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def rows(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_complete_matched_reduction() -> None:
    conditions = rows("condition-comparison.csv")
    matched = rows("matched-speedup-change.csv")
    operations = rows("operation-comparison.csv")
    attention = rows("attention-comparison.csv")
    assert len(conditions) == 12
    assert len(matched) == 6
    assert len(operations) == 36
    assert len(attention) == 4
    assert {(row["model_size"], row["condition_id"]) for row in conditions} == {
        (size, condition)
        for size in ("70M", "410M")
        for condition in (
            "a0-gelu",
            "a1h-relu",
            "a4-ol1-kappa-0",
            "a4-ol1-kappa-0p5",
            "a7-ol1-kappa-0",
            "a7-ol1-kappa-0p5",
        )
    }


def test_headline_claims_are_machine_checked() -> None:
    summary = json.loads((HERE / "comparison-summary.json").read_text(encoding="utf-8"))
    assert summary["same_physical_gpu"] is True
    assert summary["full_model_break_even_count"] == {
        "70M": {"batch_1": 0, "batch_32": 0},
        "410M": {"batch_1": 0, "batch_32": 0},
    }
    assert summary["primitive_break_even_count"] == {"70M": 0, "410M": 0}
    assert summary["attention_break_even_count"] == {"70M": 0, "410M": 0}
    assert summary["410m_improvement_count_over_70m"] == {"batch_1": 0, "batch_32": 1}
    assert all(value > 1 for value in summary["official_positive_control_speedup"].values())


def test_cross_device_attention_boundary_is_explicit() -> None:
    conditions = rows("condition-comparison.csv")
    incomparable = [
        row for row in conditions
        if row["attention_coverage_comparable_to_canonical_R_model"] == "False"
    ]
    assert [(row["model_size"], row["condition_id"]) for row in incomparable] == [
        ("410M", "a7-ol1-kappa-0p5")
    ]
    assert incomparable[0]["R_covered_linear_plus_attention"] == ""
