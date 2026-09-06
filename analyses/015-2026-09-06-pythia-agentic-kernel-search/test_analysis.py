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


def test_final_matrix_is_complete_and_gate_failures_are_explicit() -> None:
    rows = read_csv("final-results.csv")
    assert len(rows) == 36
    assert Counter(row["model_size"] for row in rows) == {"14M": 12, "70M": 12, "410M": 12}
    failures = {row["condition_id"] for row in rows if row["correctness_pass"] == "False"}
    assert failures == {"14m/a7-0p1", "70m/a4-0p01", "70m/a4-0p05", "70m/a4-0p1"}
    assert all(int(row["paired_timing_samples"]) == 80 for row in rows)
    assert all(int(row["timing_input_clusters"]) == 16 for row in rows)


def test_qualified_associations_match_frozen_reduction() -> None:
    fits = {
        row["model_size"]: row
        for row in read_csv("regressions.csv")
        if row["stratum"] == "qualified-final"
    }
    assert set(fits) == {"14M", "70M", "410M"}
    assert int(fits["14M"]["n_checkpoints"]) == 11
    assert int(fits["70M"]["n_checkpoints"]) == 9
    assert int(fits["410M"]["n_checkpoints"]) == 12
    assert float(fits["14M"]["R_squared"]) == pytest.approx(0.2351301056360091)
    assert float(fits["70M"]["R_squared"]) == pytest.approx(0.0026553630129529937)
    assert float(fits["410M"]["R_squared"]) == pytest.approx(0.6106405945041313)


def test_same_rmodel_transition_inventory() -> None:
    rows = read_csv("same-rmodel-optimization.csv")
    counts = Counter(row["comparison_group"] for row in rows)
    assert counts == {
        "14M exploratory P0-to-K001": 4,
        "14M final-policy repair K012-to-K013": 6,
        "70M complete-validation K009-to-K016": 12,
        "410M development K004-to-K010": 3,
    }
    assert all(float(row["R_model_fraction"]) >= 0 for row in rows)


def test_retrieval_identity_is_closed_with_one_declared_log_race() -> None:
    verification = json.loads((HERE / "source-verification.json").read_text(encoding="utf-8"))
    assert verification["passed"] is True
    assert verification["archive"]["bytes"] == 5_840_881
    assert verification["archive"]["sha256"] == (
        "404d02e668c6de6b9cf2e2835bef3c166e82d61c12e04c6bfb8bfcc66466777d"
    )
    assert verification["inventory"]["listed_files"] == 4_101
    assert verification["inventory"]["matching_files"] == 4_100
    assert verification["known_controller_log_race"]["path"].endswith("phase13-package-evidence-rtxpro4500-002.launch.log")


def test_reducer_is_deterministic() -> None:
    outputs = [
        HERE / "final-results.csv",
        HERE / "regressions.csv",
        HERE / "same-rmodel-optimization.csv",
        HERE / "source-verification.json",
        HERE / "reduction.json",
        HERE / "tables.md",
    ]
    before = {path.name: file_hash(path) for path in outputs}
    subprocess.run([sys.executable, str(HERE / "01_reduce.py")], check=True, cwd=HERE.parents[1])
    after = {path.name: file_hash(path) for path in outputs}
    assert after == before


@pytest.mark.parametrize(
    "name",
    ["01-rmodel-vs-full-model-speedup.pdf", "02-same-rmodel-kernel-search-transitions.pdf"],
)
def test_publication_pdf_exists(name: str) -> None:
    path = HERE / "figures" / name
    assert path.stat().st_size > 10_000
    assert path.read_bytes().startswith(b"%PDF-")
