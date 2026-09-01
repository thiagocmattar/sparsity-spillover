"""Verify Status Report Number 2's generated evidence and deliverables."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
BUILDER_PATH = REPORT_DIR / "01_build.py"
TEX_PATH = REPORT_DIR / "status-update.tex"
PDF_PATH = REPORT_DIR / "status-update.pdf"
BILLING_PATH = REPORT_DIR / "billing-refresh.json"
ANALYSIS_PATH = (
    REPO_ROOT
    / "analyses"
    / "010-2026-09-01-pythia14m-vs-70m-selected-ladder"
    / "figure_data.json"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("status_report_02_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load report builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = _load_builder()
    expected_tex = builder.build_report()
    actual_tex = TEX_PATH.read_text(encoding="utf-8")
    assert actual_tex == expected_tex, "status-update.tex is stale relative to 01_build.py"

    billing = json.loads(BILLING_PATH.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    main_rows = billing["main_report_runs"]
    assert {row["run"] for row in main_rows} == {
        "004",
        "009",
        "011",
        "013",
        "014",
        "015",
        "018",
    }
    assert math.isclose(
        sum(row["total_cost_usd"] for row in main_rows),
        billing["main_report_total_usd"],
        abs_tol=1e-10,
    )
    assert math.isclose(
        billing["main_report_total_usd"]
        + billing["historical_run_012"]["total_cost_usd"],
        billing["total_including_historical_run_012_usd"],
        abs_tol=1e-10,
    )
    assert billing["resource_audit"]["active_pod_count"] == 0
    assert len(billing["resource_audit"]["retained_network_volumes"]) == 1

    assert analysis["schema_version"] == 2
    assert analysis["status"] == "complete_verified_analysis"
    assert len(analysis["trained_endpoints"]) == 20
    assert len(analysis["teal_points"]) == 40
    assert {
        (row["family"], row["scale"], row["kappa"])
        for row in analysis["trained_endpoints"]
    } == {
        (family, scale, kappa)
        for family in ("A4-OL1", "A7-OL1")
        for scale in ("14M", "70M")
        for kappa in (0.0, 0.01, 0.05, 0.1, 0.5)
    }
    assert {
        (row["control"], row["scale"], row["target_sparsity"])
        for row in analysis["teal_points"]
    } == {
        (control, scale, target / 10)
        for control in ("A0", "A1-H")
        for scale in ("14M", "70M")
        for target in range(10)
    }

    required_tex = (
        r"\title{Status Report Number 2}",
        r"\includegraphics[width=0.80\textwidth]{../../artifacts/pythia-architecture-sparsification-ladder.pdf}",
        r"A0 & 1 & Full pass complete: Run 004 & 1/1 + TEAL: Run 018 & 0/1",
        r"A4-OL1 & 5 & 5/5 full pass: Run 015 & 5/5 full pass: Run 018 & 0/5",
        r"\label{tab:a4-ol1-scale}",
        r"\label{tab:a7-ol1-scale}",
        r"\label{tab:teal-controls-70m}",
        r"\label{fig:fullpass-frontier}",
        r"Four-family promotion complete in Analysis 010; descriptive",
        r"Current Pod spend across the seven main report runs is \$237.61",
        r"current Pod spend across all report rows is \$252.46",
    )
    for fragment in required_tex:
        assert fragment in actual_tex, f"Missing report fragment: {fragment}"

    for target in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"):
        assert f"{target} & " in actual_tex, f"Missing displayed TEAL target {target}"

    pdf = PDF_PATH.read_bytes()
    assert pdf.startswith(b"%PDF-"), "status-update.pdf is not a PDF"
    assert len(pdf) > 100_000, "status-update.pdf is unexpectedly small"
    print("verified Status Report Number 2 source, evidence coverage, billing, and PDF")


if __name__ == "__main__":
    main()
