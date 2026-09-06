from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_retrieved_process_and_declared_mode_is_accounted_for() -> None:
    rows = read_csv("replication-processes.csv")
    assert len(rows) == 256
    assert Counter(row["hardware_attempt"] for row in rows) == {
        "rtxpro4500-003": 128,
        "h100nvl-002": 128,
    }
    assert len({(row["hardware_attempt"], row["process_id"]) for row in rows}) == 220
    assert Counter(
        (row["hardware_attempt"], row["role"]) for row in rows
    ) == {
        ("rtxpro4500-003", "winner"): 108,
        ("rtxpro4500-003", "component"): 20,
        ("h100nvl-002", "winner"): 72,
        ("h100nvl-002", "p0"): 36,
        ("h100nvl-002", "component"): 20,
    }


def test_primary_process_aggregation_is_complete_and_failures_are_explicit() -> None:
    rows = read_csv("replication-summary.csv")
    assert len(rows) == 54
    assert Counter(row["hardware_attempt"] for row in rows) == {
        "rtxpro4500-003": 36,
        "h100nvl-002": 18,
    }
    rtx = [row for row in rows if row["hardware_attempt"] == "rtxpro4500-003"]
    h100 = [row for row in rows if row["hardware_attempt"] == "h100nvl-002"]
    assert all(row["n_fresh_processes"] == "3" for row in rtx)
    assert all(row["primary_process_repeats"] == "1;2;3" for row in rtx)
    assert all(row["n_fresh_processes"] == "2" for row in h100)
    assert all(row["primary_process_repeats"] == "2;3" for row in h100)
    assert all(row["qualified"] == "True" for row in h100)
    failures = {row["condition_id"] for row in rtx if row["qualified"] == "False"}
    assert failures == {
        "14m/a7-0p1",
        "70m/a4-0p01",
        "70m/a4-0p05",
        "70m/a4-0p1",
    }
    for row in rows:
        ratio = int(row["block_zero_product_count"]) / int(row["model_product_count"])
        assert ratio == pytest.approx(float(row["measured_R_model_fraction"]), abs=1e-15)


def test_fresh_process_regressions_match_the_reduction() -> None:
    fits = {
        (row["hardware_attempt"], row["model_size"]): row
        for row in read_csv("replication-regressions.csv")
        if row["stratum"] == "primary-qualified"
    }
    assert set(fits) == {
        (attempt, size)
        for attempt in ("rtxpro4500-003", "h100nvl-002")
        for size in ("14M", "70M", "410M")
    }
    expected_r2 = {
        ("rtxpro4500-003", "14M"): 0.3467936613226221,
        ("rtxpro4500-003", "70M"): 0.0027274602273558157,
        ("rtxpro4500-003", "410M"): 0.6358217182880768,
        ("h100nvl-002", "14M"): 0.32062930634179565,
        ("h100nvl-002", "70M"): 0.05107574616690558,
        ("h100nvl-002", "410M"): 0.6998781443014184,
    }
    for key, expected in expected_r2.items():
        assert float(fits[key]["R_squared"]) == pytest.approx(expected)
        assert float(fits[key]["slope_per_10_percentage_points"]) > 0
    assert int(fits[("rtxpro4500-003", "14M")]["n_checkpoints"]) == 11
    assert int(fits[("rtxpro4500-003", "70M")]["n_checkpoints"]) == 9
    assert int(fits[("rtxpro4500-003", "410M")]["n_checkpoints"]) == 12
    assert all(int(fits[("h100nvl-002", size)]["n_checkpoints"]) == 6 for size in ("14M", "70M", "410M"))


def test_each_model_size_has_a_verified_speedup_on_at_least_one_gpu() -> None:
    rows = [row for row in read_csv("replication-summary.csv") if row["qualified"] == "True"]
    by_size = {
        size: max(float(row["process_median_speedup"]) for row in rows if row["model_size"] == size)
        for size in ("14M", "70M", "410M")
    }
    assert by_size["14M"] == pytest.approx(1.072539187231941)
    assert by_size["70M"] == pytest.approx(1.0451614041367363)
    assert by_size["410M"] == pytest.approx(1.0192068242996246)
    assert all(value > 1.0 for value in by_size.values())


def test_hardware_transfer_is_matched_but_speedup_is_not_invariant() -> None:
    rows = read_csv("hardware-transfer.csv")
    assert len(rows) == 18
    assert all(row["both_qualified"] == "True" for row in rows)
    assert Counter(row["model_size"] for row in rows) == {"14M": 6, "70M": 6, "410M": 6}
    changes = [float(row["h100_minus_rtx_speedup"]) for row in rows]
    assert min(changes) < -0.08
    assert max(changes) > 0.02


def test_component_and_graph_controls_remain_separate() -> None:
    components = read_csv("component-results.csv")
    assert len(components) == 40
    assert all(row["component_correctness_pass"] == "True" for row in components)
    attention = [row for row in components if row["component"] == "attention_projection"]
    assert len(attention) == 18
    assert all(float(row["component_speedup"]) > 1.0 for row in attention)

    graphs = read_csv("cuda-graph-controls.csv")
    assert len(graphs) == 36
    assert sum(row["timing_available"] == "True" for row in graphs) == 19
    assert sum(row["qualified_complete_timing"] == "True" for row in graphs) == 14
    assert sum(
        row["role"] == "winner" and row["timing_available"] == "True" for row in graphs
    ) == 15


def test_h100_starting_point_comparison_does_not_overclaim_optimization() -> None:
    rows = read_csv("h100-same-rmodel.csv")
    assert len(rows) == 18
    timed = [row for row in rows if row["timing_pair_available"] == "True"]
    qualified = [row for row in rows if row["both_qualified"] == "True"]
    assert len(timed) == 4
    assert len(qualified) == 1
    assert qualified[0]["condition_id"] == "14m/a4-0p5"
    assert float(qualified[0]["optimized_over_baseline_ratio"]) == pytest.approx(
        0.922737344905385
    )


def test_retrieval_archives_match_closeout_identities() -> None:
    reduction = json.loads((HERE / "replication-reduction.json").read_text(encoding="utf-8"))
    assert reduction["retrievals"]["rtxpro4500-003"]["bytes"] == 6_334_093
    assert reduction["retrievals"]["rtxpro4500-003"]["sha256"] == (
        "74783671c3db7c795de20f19482d4db583868b0f26d781be2358a82cef839cbf"
    )
    assert reduction["retrievals"]["h100nvl-002"]["bytes"] == 15_267_840
    assert reduction["retrievals"]["h100nvl-002"]["sha256"] == (
        "0a38a9f425eb8bfffa287682ca77acdbc7814200e9b99e8202fa55a49a8462b9"
    )


def test_replication_reducer_is_deterministic() -> None:
    names = [
        "replication-processes.csv",
        "replication-summary.csv",
        "replication-regressions.csv",
        "hardware-transfer.csv",
        "component-results.csv",
        "cuda-graph-controls.csv",
        "h100-same-rmodel.csv",
        "replication-reduction.json",
        "replication-tables.md",
    ]
    before = {name: file_hash(HERE / name) for name in names}
    subprocess.run(
        [sys.executable, str(HERE / "03_reduce_replications.py")],
        check=True,
        cwd=HERE.parents[1],
    )
    after = {name: file_hash(HERE / name) for name in names}
    assert after == before
