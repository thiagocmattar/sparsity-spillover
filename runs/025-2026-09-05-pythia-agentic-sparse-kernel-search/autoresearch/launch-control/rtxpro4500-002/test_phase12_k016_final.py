from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "phase12-70m-k016-frozen-final.sh"


def test_phase12_runs_six_development_and_six_retrospective_conditions():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("run_probe 70m/") == 12
    assert "--inputs 16 --passes 5" in text
    assert "--full-validation" in text
    assert "70m/a0 all" in text
    assert "retrospective" in text


def test_phase12_never_changes_policy_by_kappa():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("--implementation k016") == 1
