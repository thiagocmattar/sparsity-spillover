import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "002-2026-08-29-l1n-spillover-local"
sys.path.insert(0, str(RUN_DIR))

import sitewise_figure  # noqa: E402


def _statistics(site_values):
    pooled = []
    for site, (hits, total, misleading_fraction) in site_values.items():
        pooled.append(
            {
                "name": site,
                "total": total,
                "threshold_hits": {"0.001": hits},
                "threshold_fractions": {"0.001": misleading_fraction},
            }
        )
    return {"pooled_by_site": pooled, "rows": []}


def test_pooled_counts_recomputes_fraction_from_integer_counts():
    stats = _statistics({"h": (25, 100, 0.25)})
    row = sitewise_figure._pooled_counts(stats, "h", 0.001)
    assert row == {"hits": 25, "total": 100, "fraction": 0.25, "percent": 25.0}


def test_sitewise_reduction_builds_two_five_point_series_and_four_panels():
    conditions = []
    m_conditions = []
    original = {}
    weights = [0.0, 0.1, 0.5, 1.0, 5.0]
    order = 0
    for weight_index, weight in enumerate(weights):
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
            conditions.append(
                {
                    "attempt_id": attempt_id,
                    "condition": condition,
                    "final_validation_loss": 5.0,
                }
            )
            original[attempt_id] = _statistics(
                {
                    "h": (10 + order, 100, (10 + order) / 100),
                    "q_post": (1 + order, 1000, (1 + order) / 1000),
                    "k_post": (2 + order, 1000, (2 + order) / 1000),
                    "v": (3 + order, 1000, (3 + order) / 1000),
                }
            )
            m_conditions.append(
                {
                    "attempt_id": attempt_id,
                    "statistics": _statistics({"m": (4 + order, 1000, (4 + order) / 1000)}),
                }
            )
    verification = {"status": "verified", "evidence_label": "valid", "conditions": conditions}
    m_artifact = {
        "status": "verified",
        "epsilon": 0.001,
        "conditions": m_conditions,
    }
    grouped = sitewise_figure.reduce_sitewise_points(
        verification, m_artifact, lambda attempt_id: original[attempt_id]
    )
    assert list(grouped) == ["gelu", "relu"]
    assert len(grouped["gelu"]) == len(grouped["relu"]) == 5
    assert grouped["gelu"][0]["label"] == "control"
    assert grouped["relu"][-1]["label"] == "lambda=5"
    assert tuple(grouped["gelu"][0]["near_zero"])[1:] == sitewise_figure.SITES
