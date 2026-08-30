import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = (
    ROOT / "analyses" / "002-2026-08-30-run006-vs-run007-partial-a4z-ol1"
)


def test_partial_comparison_revalidates_only_completed_matched_attempts():
    spec = importlib.util.spec_from_file_location("analysis_002", ANALYSIS_DIR / "01_compare.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = module.main()
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["evidence_status"] == "partial_completed_attempts"
    assert result["available_pair_count"] == 4
    assert result["planned_pair_count"] == 5
    assert result["matched_identity_per_condition"]["optimizer_steps"] == 581
    assert result["matched_identity_per_condition"]["training_input_tokens"] == 76_152_832
    assert result["matched_identity_per_condition"]["sequences"] == 338
    assert [row["kappa"] for row in result["rows"]] == [0.0, 0.01, 0.05, 0.1]
    assert all(
        row["paired_quality_R_model_relation"] == "tradeoff" for row in result["rows"]
    )
    assert all(row["delta_run007_minus_run006"]["R_model"] > 0 for row in result["rows"])
    assert all(row["delta_run007_minus_run006"]["R_block"] > 0 for row in result["rows"])
    assert result["unpaired_conditions"][0]["kappa"] == 0.5
    assert result["unpaired_conditions"][0]["run007"] is None
    assert [(row["kappa"], row["status"]) for row in result["failed_attempts"]] == [
        (0.01, "failed"),
        (0.5, "failed"),
        (0.5, "failed"),
    ]
    frontier = {
        (row["run"], row["kappa"])
        for row in result["joint_quality_R_model_frontier_over_available_pairs"]
    }
    assert frontier == {
        ("run006", 0.1),
        ("run007", 0.01),
        ("run007", 0.05),
        ("run007", 0.1),
    }
    assert all(
        len(source["transfer_inventory"]["sha256"]) == 64
        for source in result["run007_completed_attempt_sources"]
    )
