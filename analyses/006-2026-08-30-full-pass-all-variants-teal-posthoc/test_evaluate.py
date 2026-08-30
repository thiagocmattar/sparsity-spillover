from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("01_evaluate.py")
SPEC = spec_from_file_location("analysis006_evaluate", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_approved_source_set_contains_thirteen_new_verified_local_checkpoints():
    sources = MODULE.load_source_descriptors(verify_checkpoint_content=False)
    assert tuple(source["condition_id"] for source in sources) == MODULE.NEW_CONDITION_IDS
    assert len(sources) == 13
    assert {source["series_id"] for source in sources} == {
        "a1h_naive_l1",
        "a1h_ol1",
        "a4z_threshold",
    }
    assert all(source["checkpoint"].is_dir() for source in sources)
    assert all(source["source_final_validation_loss"] < 6.0 for source in sources)


def test_reused_controls_are_complete_and_protocol_matched():
    rows, result = MODULE.load_reused_controls()
    assert result["status"] == "complete_verified"
    assert len(rows) == 20
    assert {row["condition_id"] for row in rows} == set(MODULE.CONTROL_CONDITION_IDS)
    assert {float(row["target_sparsity"]) for row in rows} == set(
        MODULE.TARGET_SPARSITIES
    )


def test_frozen_teal_threshold_rule_preserves_zero_target_and_clips_equality():
    values = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    result = MODULE.TEAL.empirical_thresholds(
        values,
        (0.0, 0.2, 0.4, 0.6, 1.0),
        np=np,
    )
    assert [row["threshold"] for row in result["targets"]] == [0.0, 0.0, 0.0, 1.0, 3.0]
    assert [row["calibration_zero_count"] for row in result["targets"]] == [2, 2, 2, 3, 5]


def test_nondominance_uses_lower_loss_and_higher_r_model():
    def row(loss, opportunity):
        return {
            "validation": {"loss": loss},
            "logical_products": {"R_model": opportunity},
        }

    rows = [row(5.0, 0.1), row(5.1, 0.2), row(5.2, 0.15), row(4.9, 0.1)]
    assert MODULE.TEAL.nondominated(rows) == [False, True, False, True]
