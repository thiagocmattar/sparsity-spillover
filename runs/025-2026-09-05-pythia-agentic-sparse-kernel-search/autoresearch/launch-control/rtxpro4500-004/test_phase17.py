from pathlib import Path


SCRIPT = Path(__file__).with_name("phase17-fixed-rmodel-pairs.sh").read_text()


def pair_rows() -> list[list[str]]:
    matrix = SCRIPT.split("pairs=(", 1)[1].split("\n)", 1)[0]
    return [line.strip().strip('"').split() for line in matrix.splitlines() if line.strip()]


def test_exact_matched_pairs() -> None:
    rows = pair_rows()
    assert len(rows) == 6
    assert [row[0] for row in rows] == [
        "14m/a4-0p5", "14m/a7-0p5",
        "70m/a4-0p5", "70m/a7-0p5",
        "410m/a4-0p5", "410m/a7-0p5",
    ]
    assert {(row[1], row[4]) for row in rows} == {
        ("p0", "k013"), ("k009", "k016"), ("k004", "k010")
    }
    assert all(row[2] == row[5] == "active" for row in rows)
    assert all(
        row[3] == "probe_k004_flagged_z_models.py"
        for row in rows if row[0].startswith("410m/")
    )


def test_three_fresh_eager_processes_per_implementation() -> None:
    assert "for run025_repeat in 1 2 3" in SCRIPT
    assert "--execution eager" in SCRIPT
    assert "--inputs 16 --passes 5" in SCRIPT
    assert "--full-validation" in SCRIPT
    assert "graph" not in SCRIPT
    assert len(pair_rows()) * 2 * 3 == 36


def test_order_is_counterbalanced_without_concurrent_timing() -> None:
    assert 'if [[ "$run025_repeat" == 2 ]]' in SCRIPT
    assert '"$run025_winner"' in SCRIPT
    assert '"$run025_baseline"' in SCRIPT
    assert not any(line.rstrip().endswith(" &") for line in SCRIPT.splitlines())
    assert "process-order.tsv" in SCRIPT


def test_processes_are_bounded_and_failures_are_retained() -> None:
    assert "timeout --signal=TERM --kill-after=30s 600s" in SCRIPT
    assert "set +e" in SCRIPT
    assert "$run025_label.exit-code.txt" in SCRIPT
    assert "$run025_label.finished-utc.txt" in SCRIPT
