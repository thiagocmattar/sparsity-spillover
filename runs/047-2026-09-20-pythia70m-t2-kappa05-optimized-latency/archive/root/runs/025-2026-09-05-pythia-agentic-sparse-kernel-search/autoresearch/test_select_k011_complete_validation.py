from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "select_k011_complete_validation.py"
SPEC = importlib.util.spec_from_file_location("run025_select_k011_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(mask, score, layers, eligible=True):
    return {
        "mask": mask,
        "primary_geometric_mean_speedup": score,
        "selected_layers": layers,
        "eligible": eligible,
    }


def test_geometric_mean_and_no_eligible_case():
    assert MODULE.geometric_mean([1.0, 4.0]) == 2.0
    assert MODULE.choose([row("a", 2.0, 1, False)]) is None


def test_selection_uses_score_then_predeclared_tie_break():
    rows = [
        row("suffix4", 1.100, 4),
        row("suffix2", 1.096, 2),
        row("suffix1", 1.094, 1),
    ]
    # suffix2 is within 0.005 of the best and has fewer layers; suffix1 is not.
    assert MODULE.choose(rows)["mask"] == "suffix2"
    rows = [row("odd", 1.100, 3), row("suffix3", 1.100, 3)]
    assert MODULE.choose(rows)["mask"] == "odd"


def test_primary_score_excludes_controls():
    assert MODULE.PRIMARY == (
        "14m/a4-0",
        "14m/a4-0p5",
        "14m/a7-0",
        "14m/a7-0p5",
    )


def test_resolver_prefers_only_complete_attempt(tmp_path):
    original = tmp_path / "k011full-14m-a7-0-suffix3-rtxpro4500-002"
    retry = tmp_path / "k011retry-14m-a7-0-suffix3-rtxpro4500-002"
    original.mkdir()
    retry.mkdir()
    (original / "status.json").write_text('{"stage":"development_quality"}')
    (retry / "status.json").write_text('{"stage":"complete"}')
    selected, abandoned = MODULE.resolve_directory(tmp_path, "a7-0", "suffix3")
    assert selected == retry
    assert abandoned == [original.name]
