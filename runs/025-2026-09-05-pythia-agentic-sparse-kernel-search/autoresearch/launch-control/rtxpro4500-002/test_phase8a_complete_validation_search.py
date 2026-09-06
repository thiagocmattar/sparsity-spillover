from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "phase8a-14m-k011-complete-validation-mask-search.sh"


def test_phase8a_is_development_only_and_full_validation():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--full-validation" in text
    assert "--inputs 16 --passes 3" in text
    assert "14m/a4-0p01" not in text
    assert "14m/a7-0p01" not in text
    for condition in ("14m/a0", "14m/a1h", "14m/a4-0", "14m/a4-0p5", "14m/a7-0", "14m/a7-0p5"):
        assert condition in text
    assert "screen-failed-confirmation-skipped" in text
    assert "all-six-passed" in text


def test_phase8a_uses_exact_predeclared_shortlist_order():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "for run025_mask in prefix1 suffix1 suffix2 suffix3 suffix4 odd" in text
