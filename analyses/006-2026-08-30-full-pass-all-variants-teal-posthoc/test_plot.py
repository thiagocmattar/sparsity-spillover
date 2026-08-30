from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


SCRIPT = Path(__file__).with_name("02_plot.py")
SPEC = spec_from_file_location("analysis006_plot", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _synthetic_result(comparison):
    rows = []
    model_products = 1_000_000_000
    operation_products = 500_000_000
    for condition_index, condition in enumerate(comparison["conditions"]):
        baseline_loss = float(condition["final_validation_loss"]) + 1e-5
        for target_index, target in enumerate(MODULE.TARGETS):
            loss = baseline_loss + 0.11 * target_index
            numerator = int(
                model_products * min(0.99, float(condition["R_model"]) + 0.01 * target_index)
            )
            rows.append(
                {
                    "condition_id": condition["condition_id"],
                    "target_sparsity": target,
                    "evidence_origin": "synthetic-test",
                    "loss_delta_from_zero_threshold": loss - baseline_loss,
                    "validation": {
                        "loss": loss,
                        "sequences": 338,
                        "input_tokens": 692_224,
                        "source_tokens": 693_668,
                        "excluded_tail_tokens": 1_444,
                        "complete_block_coverage": True,
                    },
                    "logical_products": {
                        "R_model": numerator / model_products,
                        "block_zero_product_count": numerator,
                        "block_product_count": operation_products,
                        "model_product_count": model_products,
                        "per_operation": {
                            "synthetic": {
                                "zero_product_count": numerator,
                                "product_count": operation_products,
                            }
                        },
                    },
                }
            )
    return {
        "schema_version": 1,
        "status": "complete_verified",
        "coverage": {
            "documents": 500,
            "sequences": 338,
            "input_tokens": 692_224,
            "source_tokens": 693_668,
            "excluded_tail_tokens": 1_444,
            "complete_block_coverage": True,
            "seed_count": 1,
        },
        "conditions": rows,
    }


def test_figure_data_contains_all_curves_and_uses_canonical_anchors():
    comparison = json.loads(MODULE.COMPARISON_PATH.read_text(encoding="utf-8"))
    data = MODULE.build_figure_data(_synthetic_result(comparison), comparison)
    assert data["display"] == {
        "y_min": 5.075,
        "y_max": 6.0,
        "rule": "Points above loss 6 are omitted rather than clipped to the boundary.",
        "layout": "single_panel",
    }
    assert data["counts"]["trained_endpoints"] == 15
    assert data["counts"]["posthoc_curves"] == 15
    assert data["counts"]["posthoc_points"] == 150
    endpoints = {row["condition_id"]: row for row in comparison["conditions"]}
    for curve in data["posthoc_curves"]:
        all_points = curve["visible_points"] + curve["omitted_above_y_cap"]
        assert len(all_points) == 10
        zero = next(row for row in all_points if row["target_sparsity"] == 0.0)
        canonical = endpoints[curve["condition_id"]]
        assert zero["R_model"] == canonical["R_model"]
        assert zero["final_validation_loss"] == canonical["final_validation_loss"]
        assert all(row["final_validation_loss"] <= 6.0 for row in curve["visible_points"])
        assert all(
            row["final_validation_loss"] > 6.0
            for row in curve["omitted_above_y_cap"]
        )
    assert data["teal_augmented_nondominated_envelope"]
