import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analyses" / "001-2026-08-30-run008-vs-run010-all-site-ol1"


def test_generated_comparison_is_matched_and_reproducible():
    spec = importlib.util.spec_from_file_location("analysis_001", ANALYSIS_DIR / "01_compare.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = module.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["matched_identity"]["total_optimizer_steps"] == 2905
    assert result["matched_identity"]["total_training_input_tokens"] == 380_764_160
    assert [row["kappa"] for row in result["rows"]] == [0.0, 0.01, 0.05, 0.1, 0.5]
    assert [row["paired_quality_R_model_relation"] for row in result["rows"]] == [
        "tradeoff", "run010_dominates", "tradeoff", "tradeoff", "run010_dominates"
    ]
    frontier = {(row["run"], row["kappa"]) for row in result["joint_quality_R_model_frontier"]}
    assert frontier == {
        ("run010", 0.01),
        ("run008", 0.1),
        ("run010", 0.05),
        ("run010", 0.1),
        ("run010", 0.5),
    }
    assert all(len(source["sha256"]) == 64 for source in result["source_files"].values())
