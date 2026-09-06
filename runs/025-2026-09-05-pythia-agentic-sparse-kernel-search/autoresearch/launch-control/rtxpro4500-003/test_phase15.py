from pathlib import Path


SCRIPT = Path(__file__).with_name("phase15-fresh-process-and-components.sh").read_text()


def test_three_fresh_processes_cover_the_complete_matrix() -> None:
    assert "for run025_repeat in 1 2 3" in SCRIPT
    matrix = SCRIPT.split("matrix=(", 1)[1].split("\n)", 1)[0]
    rows = [line.strip().strip('"') for line in matrix.splitlines() if line.strip()]
    assert len(rows) == 36
    assert sum(row.startswith("14m/") for row in rows) == 12
    assert sum(row.startswith("70m/") for row in rows) == 12
    assert sum(row.startswith("410m/") for row in rows) == 12
    assert len({row.split()[0] for row in rows}) == 36


def test_component_probes_are_separate_complete_validation_processes() -> None:
    assert "probe_frozen_ffn.py" in SCRIPT
    assert "probe_frozen_attention_projection.py" in SCRIPT
    assert "--full-validation" in SCRIPT
    assert "QK" not in SCRIPT


def test_worker_is_bounded_and_durable() -> None:
    assert "timeout --signal=TERM --kill-after=30s 600s" in SCRIPT
    assert "exit-code.txt" in SCRIPT
    assert "finished-utc.txt" in SCRIPT
