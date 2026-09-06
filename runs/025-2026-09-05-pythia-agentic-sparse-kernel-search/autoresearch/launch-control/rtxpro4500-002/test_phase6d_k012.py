from pathlib import Path
import re


SCRIPT = Path(__file__).with_name("phase6d-14m-k012-six-development.sh")


def test_phase6d_has_exact_six_development_conditions():
    source = SCRIPT.read_text(encoding="utf-8")
    calls = re.findall(r"^run_probe (14m/\S+) (all|h|active) (\S+)$", source, re.MULTILINE)
    assert [row[0] for row in calls] == [
        "14m/a0", "14m/a1h", "14m/a4-0", "14m/a4-0p5",
        "14m/a7-0", "14m/a7-0p5",
    ]
    assert "--inputs 16 --passes 10" in source
    assert "0p01" not in source and "0p05" not in source
