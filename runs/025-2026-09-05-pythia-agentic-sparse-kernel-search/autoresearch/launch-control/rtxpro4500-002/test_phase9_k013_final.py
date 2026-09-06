from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "phase9-14m-k013-frozen-final.sh"


def test_phase9_requires_frozen_complete_validation_before_heldout():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "probe_k013_final.py" in text
    assert "--implementation k013" in text
    assert "--inputs 16 --passes 5" in text
    assert "--full-validation" in text
    assert "if [ \"$run025_development_failed\" -ne 0 ]" in text
    assert text.index("run_probe 14m/a7-0p5") < text.index("run_probe 14m/a4-0p01")
    assert "failed-heldout-skipped" in text


def test_phase9_has_exact_six_plus_six_condition_matrix():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("run_probe 14m/") == 12
    for suffix in ("0p01", "0p05", "0p1"):
        assert f"14m/a4-{suffix}" in text
        assert f"14m/a7-{suffix}" in text
