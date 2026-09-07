from pathlib import Path
import importlib.util


SOURCE = Path(__file__).resolve().parent / "select_k014_complete_validation.py"
SPEC = importlib.util.spec_from_file_location("run025_select_k014_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(mask, score, layers, eligible=True):
    return {"mask": mask, "score": score, "selected_layers": layers, "eligible": eligible}


def test_geometric_mean_and_no_winner():
    assert MODULE.geometric_mean([1.0, 4.0]) == 2.0
    assert MODULE.choose([row("all6", 2.0, 6, False)]) is None


def test_tie_break_prefers_fewer_layers_within_tolerance():
    rows = [row("all6", 1.100, 6), row("suffix3", 1.096, 3), row("suffix2", 1.094, 2)]
    assert MODULE.choose(rows)["mask"] == "suffix3"


def test_selection_groups_reuse_one_mask_across_kappas():
    assert MODULE.CONDITIONS["A4-OL1"] == ("70m/a4-0", "70m/a4-0p5")
    assert MODULE.CONDITIONS["A7-OL1"] == ("70m/a7-0", "70m/a7-0p5")
