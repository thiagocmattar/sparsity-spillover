from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "phase11-70m-k015-a7-site-search.sh"


def test_phase11_is_bounded_a7_development_only():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--full-validation" in text
    assert "--inputs 16 --passes 3" in text
    assert "suffix3-hz suffix4-hz suffix5-hz odd-hz all-h all-z all-hz all-ahz all-mhz" in text
    assert "70m/a7-0" in text and "70m/a7-0p5" in text
    assert "70m/a7-0p01" not in text and "70m/a7-0p05" not in text
