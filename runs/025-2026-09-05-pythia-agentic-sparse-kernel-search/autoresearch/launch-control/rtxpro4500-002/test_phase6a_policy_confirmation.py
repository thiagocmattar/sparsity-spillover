from pathlib import Path
import re


SCRIPT = Path(__file__).with_name("phase6a-14m-70m-policy-confirmation.sh")


def test_phase6a_uses_only_predeclared_development_conditions():
    source = SCRIPT.read_text(encoding="utf-8")
    calls = re.findall(
        r"^run_probe (\S+) (14m|70m)/(\S+) (k001|k003|k009) (\S+) (\S+)$",
        source,
        flags=re.MULTILINE,
    )
    assert len(calls) == 12
    assert [f"{size}/{condition}" for _, size, condition, _, _, _ in calls[:6]] == [
        "14m/a0", "14m/a1h", "14m/a4-0", "14m/a4-0p5",
        "14m/a7-0", "14m/a7-0p5",
    ]
    assert {f"{size}/{condition}" for _, size, condition, _, _, _ in calls[6:]} == {
        "70m/a4-0p5", "70m/a7-0p5",
    }
    assert "--inputs 16 --passes 10" in source
    assert "--full-validation" not in source
    assert "0p01" not in source and "0p05" not in source and "0p1 " not in source
