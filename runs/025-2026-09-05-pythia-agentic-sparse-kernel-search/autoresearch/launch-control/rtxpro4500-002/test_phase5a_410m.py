from pathlib import Path
import re


SCRIPT = Path(__file__).with_name("phase5a-410m-three-endpoint-baselines.sh")


def test_phase5a_is_fixed_to_three_implementations_and_three_endpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "for run025_implementation in p0 k001 k004" in source
    assert source.count("for run025_implementation in p0 k001 k004") == 3
    assert set(re.findall(r"410m/(a0|a4-0p5|a7-0p5)", source)) == {
        "a0",
        "a4-0p5",
        "a7-0p5",
    }
    assert "--inputs 16 --passes 3" in source


def test_phase5a_does_not_touch_heldout_or_full_validation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "full-validation" not in source
    assert "heldout" not in source
    assert "410m/a4-0 " not in source
    assert "410m/a7-0 " not in source
