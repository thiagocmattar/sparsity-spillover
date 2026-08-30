import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "002-2026-08-29-l1n-spillover-local"
sys.path.insert(0, str(RUN_DIR))

import eps1e2_figures  # noqa: E402


def _statistics(site_values):
    return {
        "pooled_by_site": [
            {
                "name": site,
                "total": total,
                "threshold_hits": {"0.01": hits},
                "threshold_fractions": {"0.01": hits / total},
            }
            for site, (hits, total) in site_values.items()
        ],
        "rows": [],
    }


def test_epsilon_1e2_pooled_counts_use_integer_hits():
    row = eps1e2_figures._pooled_counts(_statistics({"h": (25, 100)}), "h")
    assert row == {"hits": 25, "total": 100, "fraction": 0.25, "percent": 25.0}


def test_reduction_builds_all_three_estimands_for_two_five_point_lines():
    conditions = []
    m_rows = []
    output_rows = []
    original = {}
    order = 0
    for weight_index, weight in enumerate((0.0, 0.1, 0.5, 1.0, 5.0)):
        for activation in ("gelu", "relu"):
            order += 1
            attempt_id = f"attempt-{order}"
            control = weight_index == 0
            condition = {
                "id": f"{activation}-{'control' if control else weight}",
                "activation": activation,
                "order": order,
                "is_control": control,
                "pressure_weight": weight,
            }
            conditions.append({"attempt_id": attempt_id, "condition": condition, "final_validation_loss": 5.0})
            original[attempt_id] = _statistics(
                {"h": (10 + order, 100), "q_post": (10, 1000), "k_post": (20, 1000), "v": (30, 1000)}
            )
            m_rows.append({"attempt_id": attempt_id, "statistics": _statistics({"m": (40, 1000)})})
            output_rows.append(
                {"attempt_id": attempt_id, "statistics": _statistics({"attention_output": (50, 1000)})}
            )
    verification = {"status": "verified", "evidence_label": "valid", "conditions": conditions}
    m_diagnostic = {"status": "verified", "conditions": m_rows}
    output_diagnostic = {"status": "verified", "conditions": output_rows}
    grouped = eps1e2_figures.reduce_points(
        verification, m_diagnostic, output_diagnostic, lambda attempt_id: original[attempt_id]
    )
    assert len(grouped["gelu"]) == len(grouped["relu"]) == 5
    point = grouped["gelu"][0]
    assert point["attention_mean"]["percent"] == 2.0
    assert point["near_zero"]["m"]["percent"] == 4.0
    assert point["near_zero"]["attention_output"]["percent"] == 5.0
    assert [path.name[:2] for path in eps1e2_figures.FIGURE_PATHS.values()] == ["04", "05", "06"]
