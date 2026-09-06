from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "phase10-70m-k014-complete-validation-search.sh"


def test_phase10_is_bounded_development_only_complete_validation():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--full-validation" in text
    assert "--inputs 16 --passes 3" in text
    assert "for run025_mask in suffix2 suffix3 suffix4 suffix5 all6 odd" in text
    for condition in ("70m/a1h", "70m/a4-0", "70m/a4-0p5", "70m/a7-0", "70m/a7-0p5"):
        assert condition in text
    for forbidden in ("70m/a4-0p01", "70m/a4-0p05", "70m/a7-0p01", "70m/a7-0p05"):
        assert forbidden not in text


def test_phase10_records_each_attempt_and_selection():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "started-utc.txt" in text
    assert "exit-code.txt" in text
    assert "finished-utc.txt" in text
    assert "select_k014_complete_validation.py" in text
