from pathlib import Path


SCRIPT = Path(__file__).with_name("phase18-14m-k001-confirmation.sh").read_text()


def test_direct_p0_to_k001_scope_is_exact() -> None:
    conditions = SCRIPT.split("conditions=(", 1)[1].split("\n)", 1)[0]
    rows = [line.strip().strip('"') for line in conditions.splitlines() if line.strip()]
    assert rows == ["14m/a4-0p5 14m-a4-0p5", "14m/a7-0p5 14m-a7-0p5"]
    assert "for run025_repeat in 1 2 3" in SCRIPT
    assert "k001" in SCRIPT and "p0" in SCRIPT
    assert len(rows) * 2 * 3 == 12


def test_phase18_waits_for_immutable_phase17_completion() -> None:
    assert "phase17-fixed-rmodel-pairs-rtxpro4500-004" in SCRIPT
    assert 'test "$(cat "$run025_phase17/exit-code.txt")" = 0' in SCRIPT
    assert "--execution eager" in SCRIPT
    assert "--inputs 16 --passes 5" in SCRIPT
    assert "--full-validation" in SCRIPT


def test_phase18_is_counterbalanced_and_bounded() -> None:
    assert 'if [[ "$run025_repeat" == 2 ]]' in SCRIPT
    assert "timeout --signal=TERM --kill-after=30s 600s" in SCRIPT
    assert not any(line.rstrip().endswith(" &") for line in SCRIPT.splitlines())
