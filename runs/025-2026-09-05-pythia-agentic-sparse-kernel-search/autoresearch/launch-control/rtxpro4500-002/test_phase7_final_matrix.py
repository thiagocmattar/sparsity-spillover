from pathlib import Path
import re


SCRIPT = Path(__file__).with_name("phase7-frozen-final-matrix.sh")


def test_phase7_has_exact_final_condition_partitioning_and_gates():
    source = SCRIPT.read_text(encoding="utf-8")
    calls = re.findall(
        r"^  run_probe ((14m|70m|410m)/\S+) (k009|k010|k012) (all|h|active) (\S+)",
        source,
        re.MULTILINE,
    )
    conditions = [row[0] for row in calls]
    assert len(conditions) == len(set(conditions)) == 30
    assert sum(item.startswith("14m/") for item in conditions) == 12
    assert sum(item.startswith("70m/") for item in conditions) == 12
    assert sum(item.startswith("410m/") for item in conditions) == 6
    assert not any(item in conditions for item in (
        "410m/a4-0p01", "410m/a4-0p05", "410m/a4-0p1",
        "410m/a7-0p01", "410m/a7-0p05", "410m/a7-0p1",
    ))
    assert "if run025_group_14m_development; then" in source
    assert "if run025_group_70m_development; then" in source
    assert "failed-heldout-skipped" in source
    assert "--inputs 16 --passes 5" in source
    assert "--full-validation" in source
