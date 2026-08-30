from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).with_name("03_plot_with_full_pass_frontier.py")
SPEC = spec_from_file_location("analysis005_plot_full_frontier", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_combined_figure_data_anchors_posthoc_curves_to_analysis004_controls():
    data = MODULE.build_figure_data()
    assert data["status"] == "complete_verified_figure_data"
    assert len(data["trained_endpoints"]) == 15
    assert data["display"]["y_max"] == 6.0

    expected = {
        "gelu-control": ([0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9]),
        "relu-control": ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]),
    }
    controls = {
        row["condition_id"]: row
        for row in data["trained_endpoints"]
        if row["condition_id"] in expected
    }
    for condition_id, (visible_targets, omitted_targets) in expected.items():
        curve = data["posthoc_control_curves"][condition_id]
        visible = curve["visible_points"]
        omitted = curve["omitted_above_y_cap"]
        assert [row["target_sparsity"] for row in visible] == visible_targets
        assert [row["target_sparsity"] for row in omitted] == omitted_targets
        assert visible[0]["R_model"] == controls[condition_id]["R_model"]
        assert visible[0]["final_validation_loss"] == controls[condition_id][
            "final_validation_loss"
        ]
        assert all(row["final_validation_loss"] <= 6.0 for row in visible)
        assert all(row["final_validation_loss"] > 6.0 for row in omitted)
