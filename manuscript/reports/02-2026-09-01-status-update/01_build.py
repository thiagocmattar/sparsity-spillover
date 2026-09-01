"""Build Status Report Number 2 from Report 1 and verified current evidence."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
BASE_TEX = (
    REPO_ROOT
    / "manuscript"
    / "reports"
    / "01-2026-08-30-status-update"
    / "status-update.tex"
)
ANALYSIS_DIR = REPO_ROOT / "analyses" / "010-2026-09-01-pythia14m-vs-70m-selected-ladder"
ANALYSIS_DATA = ANALYSIS_DIR / "figure_data.json"
BILLING_DATA = REPORT_DIR / "billing-refresh.json"
OUTPUT_TEX = REPORT_DIR / "status-update.tex"

SITES = ("h", "m", "a", "z", "q_post", "k_post", "v")
KAPPAS = (0.0, 0.01, 0.05, 0.1, 0.5)
TEAL_TARGETS = tuple(index / 10 for index in range(10))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {label} replacement target")
    return text.replace(old, new)


def _percentage(fraction: float, digits: int = 3) -> str:
    value = 100.0 * fraction
    if value == 0:
        return f"{value:.{digits}f}"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.{digits}f}"


def _family_rows(data: dict[str, Any], family: str) -> list[dict[str, Any]]:
    rows = [row for row in data["trained_endpoints"] if row["family"] == family]
    rows.sort(key=lambda row: (row["kappa"], row["scale"]))
    if len(rows) != 10:
        raise ValueError(f"Expected ten {family} rows")
    for kappa in KAPPAS:
        matched = [row for row in rows if row["kappa"] == kappa]
        if [row["scale"] for row in matched] != ["14M", "70M"]:
            raise ValueError(f"Incomplete {family} scale pair at kappa={kappa}")
        for row in matched:
            if set(row["site_exact_zero"]) != {
                "a",
                "m",
                "h",
                "q_post",
                "k_post",
                "v",
                "z",
                "attention_output",
            }:
                raise ValueError(f"Incomplete site coverage for {row['condition_id']}")
    return rows


def _family_table(data: dict[str, Any], family: str, label: str) -> str:
    rows = _family_rows(data, family)
    rendered: list[str] = []
    for index, kappa in enumerate(KAPPAS):
        pair = [row for row in rows if row["kappa"] == kappa]
        for pair_index, row in enumerate(pair):
            zeros = row["site_exact_zero"]
            first = rf"\multirow{{2}}{{*}}{{{kappa:.2f}}}" if pair_index == 0 else " " * 22
            values = [
                first,
                row["scale"],
                f"{row['validation_loss']:.6f}",
                _percentage(row["R_model"], 6),
                *[_percentage(zeros[site]["exact_zero_fraction"]) for site in SITES],
            ]
            rendered.append(" & ".join(values) + r" \\")
        if index < len(KAPPAS) - 1:
            rendered.append(r"\hline")

    return rf"""{{\scriptsize
\begin{{table}}[htbp]
\centering
\caption{{Pythia-14M and 70M {family} full-pass endpoints from \href{{../../../analyses/010-2026-09-01-pythia14m-vs-70m-selected-ladder/observations/O001-pythia14m-vs-70m-selected-ladder.md}}{{Analysis 010, Observation O001}}. Percentages pool integer counts over all six layers and 338 complete validation blocks.}}
\label{{{label}}}
\setlength{{\tabcolsep}}{{1.6pt}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}rl*{{9}}{{r}}@{{}}}}
\toprule
$\boldsymbol{{\kappa}}$ & \textbf{{Scale}} & \textbf{{Loss}} & \textbf{{\Rmodel{{}} (\%)}} & \multicolumn{{7}}{{c}}{{\textbf{{Exact-zero mass (\%)}}}} \\
\cmidrule(l){{5-11}}
& & & & \textbf{{\code{{h}}}} & \textbf{{\code{{m}}}} & \textbf{{\code{{a}}}} & \textbf{{\code{{z}}}} & \textbf{{\code{{q}}}} & \textbf{{\code{{k}}}} & \textbf{{\code{{v}}}} \\
\midrule
{chr(10).join(rendered)}
\bottomrule
\end{{tabular*}}
\end{{table}}}}"""


def _teal_70_table(data: dict[str, Any]) -> str:
    by_control = {
        control: {
            row["target_sparsity"]: row
            for row in data["teal_points"]
            if row["scale"] == "70M" and row["control"] == control
        }
        for control in ("A0", "A1-H")
    }
    for control, rows in by_control.items():
        if tuple(sorted(rows)) != tuple(index / 10 for index in range(10)):
            raise ValueError(f"Incomplete 70M {control} TEAL grid")
    rendered = []
    for target in TEAL_TARGETS:
        values = []
        for control in ("A0", "A1-H"):
            row = by_control[control][target]
            baseline = by_control[control][0.0]["validation_loss"]
            values.extend(
                [
                    f"{row['validation_loss']:.6f}",
                    f"{row['validation_loss'] - baseline:+.6f}",
                    _percentage(row["R_model"], 6),
                ]
            )
        rendered.append(f"{target:.1f} & " + " & ".join(values) + r" \\")

    return rf"""{{\small
\begin{{table}}[htbp]
\centering
\caption{{Complete Pythia-70M uniform TEAL-style post-hoc frontiers from Analysis 010. Deltas are paired to the independently evaluated $p=0$ row for the same checkpoint.}}
\label{{tab:teal-controls-70m}}
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}r*{{6}}{{r}}@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{\textbf{{A0 / GeLU}}}} & \multicolumn{{3}}{{c}}{{\textbf{{A1-H / ReLU}}}} \\
\cmidrule(lr){{2-4}}\cmidrule(l){{5-7}}
$\boldsymbol{{p}}$ & \textbf{{Loss}} & $\boldsymbol{{\Delta}}$\textbf{{loss}} & \textbf{{\Rmodel{{}} (\%)}} & \textbf{{Loss}} & $\boldsymbol{{\Delta}}$\textbf{{loss}} & \textbf{{\Rmodel{{}} (\%)}} \\
\midrule
{chr(10).join(rendered)}
\bottomrule
\end{{tabular*}}
\end{{table}}}}"""


def _run018_wall_clock_minutes() -> tuple[int, int]:
    attempts = (
        REPO_ROOT
        / "runs"
        / "018-2026-09-01-pythia70m-selected-ladder-canonical-init"
        / "artifacts"
        / "attempts"
    )
    durations = []
    for manifest_path in sorted(attempts.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        started = datetime.fromisoformat(manifest["started_at"])
        finished = datetime.fromisoformat(manifest["finished_at"])
        durations.append((finished - started).total_seconds() / 60.0)
    if len(durations) != 12:
        raise ValueError("Expected 12 completed Run 018 manifests")
    return round(min(durations)), round(max(durations))


def _compute_section(billing: dict[str, Any]) -> str:
    records = {row["run"]: row for row in billing["main_report_runs"]}
    expected = {"004", "009", "011", "013", "014", "015", "018"}
    if set(records) != expected:
        raise ValueError("Billing run set does not match the main report")
    total = sum(row["total_cost_usd"] for row in records.values())
    if not math.isclose(total, billing["main_report_total_usd"], abs_tol=1e-10):
        raise ValueError("Main billing total does not reconcile")
    run018_min, run018_max = _run018_wall_clock_minutes()
    row_specs = (
        ("004", r"$5\times$ A100 80 GB; six conditions", "63--108 min"),
        ("009", r"$4\times$ A100 80 GB; four conditions", "86--105 min"),
        ("011", r"$5\times$ A100 80 GB; five conditions", "59--104 min"),
        ("015", r"$5\times$ A100 80 GB; five conditions", "94--101 min"),
        ("013", r"$5\times$ A100 80 GB; five conditions", "82--89 min"),
        ("014", r"$5\times$ A100 80 GB; five conditions", "74--115 min"),
        (
            "018",
            "H200 141 GB; 12 conditions; 15 total Pods",
            f"{run018_min}--{run018_max} min",
        ),
    )
    rows = [
        f"{run} & {compute} & {wall_clock} & \\${records[run]['total_cost_usd']:.2f} \\\\"
        for run, compute, wall_clock in row_specs
    ]
    queried = billing["queried_at_utc"][:16].replace("T", " ")
    return rf"""\section{{RunPod compute}}

Table~\ref{{tab:runpod}} reports elapsed wall-clock time for each accepted condition's scientific process. A RunPod REST v2 refresh at {queried} UTC reconciled every listed cost to the exact Pod IDs. Current Pod spend across the seven main report runs is \${total:.2f}; the retained 100 GB network volume remains separate at \$0.01/hour (about \$7/month).

{{\small
\begin{{center}}
\begin{{minipage}}{{\textwidth}}
\captionof{{table}}{{RunPod execution record. Costs include GPU and temporary Pod-disk charges for every identified preflight, accepted worker, and infrastructure retry.}}
\label{{tab:runpod}}
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}p{{12mm}}p{{69mm}}Y r@{{}}}}
\toprule
\textbf{{Run}} & \textbf{{Compute}} & \textbf{{Wall-clock}} & \textbf{{Cost}} \\
\midrule
{chr(10).join(rows)}
\midrule
\textbf{{Total}} & \textbf{{Current Pod spend}} & --- & \textbf{{\${total:.2f}}} \\
\bottomrule
\end{{tabularx}}
\end{{minipage}}
\end{{center}}}}

The refresh supersedes earlier closeout snapshots whose last hourly buckets had not all posted, including values previously described as settled. It found zero Pods and one intentionally retained 100 GB Standard volume. Run 018 accounts for 15 H200 allocations: one timing preflight, four sentinel Pods, eight planned remainder Pods, and two infrastructure retries. Report-local \code{{billing-refresh.json}} retains the query window, Pod IDs, exact GPU/disk amounts, and reconciliation totals.

"""


def _scale_section(data: dict[str, Any]) -> str:
    checks = data["persistence_checks"]
    for scale in ("14M", "70M"):
        if not checks[scale]["A4_dominates_A7_at_kappa_0"]:
            raise ValueError(f"A4-OL1 does not dominate A7-OL1 at kappa 0 for {scale}")
        if not checks[scale]["A7_dominates_A4_at_kappa_0p5"]:
            raise ValueError(f"A7-OL1 does not dominate A4-OL1 at kappa 0.5 for {scale}")
    endpoints = {
        (row["family"], row["scale"], row["kappa"]): row
        for row in data["trained_endpoints"]
    }

    def r_gain(family: str, scale: str, lower: float, upper: float) -> float:
        return 100.0 * (
            endpoints[(family, scale, upper)]["R_model"]
            - endpoints[(family, scale, lower)]["R_model"]
        )

    def loss_delta(family: str, scale: str, lower: float, upper: float) -> float:
        return (
            endpoints[(family, scale, upper)]["validation_loss"]
            - endpoints[(family, scale, lower)]["validation_loss"]
        )

    a7_gain_14 = r_gain("A7-OL1", "14M", 0.0, 0.1)
    a7_gain_70 = r_gain("A7-OL1", "70M", 0.0, 0.1)
    a7_loss_14 = loss_delta("A7-OL1", "14M", 0.0, 0.1)
    a7_loss_70 = loss_delta("A7-OL1", "70M", 0.0, 0.1)
    a4_span_14 = r_gain("A4-OL1", "14M", 0.0, 0.5)
    a4_span_70 = r_gain("A4-OL1", "70M", 0.0, 0.5)
    a7_span_14 = r_gain("A7-OL1", "14M", 0.0, 0.5)
    a7_span_70 = r_gain("A7-OL1", "70M", 0.0, 0.5)
    return rf"""\subsection{{Selected scale promotion: Pythia-14M to 70M}}

Run 018 promotes A0, A1-H, A4-OL1, and A7-OL1 to randomly initialized Pythia-70M under the matched one-pass contract. Analysis 010 compares those endpoints with their Pythia-14M counterparts. This is a selected ladder promotion, not a complete 70M rerun of every intermediate ablation and not a scaling-law estimate.

The qualitative crossover persists. A4-OL1 dominates A7-OL1 at $\kappa=0$ at both sizes, while A7-OL1 dominates A4-OL1 at $\kappa=0.5$. Within A7-OL1, the $\kappa=0.1$ point gains {a7_gain_14:.3f} percentage points of \Rmodel{{}} over $\kappa=0$ while lowering 14M loss by {abs(a7_loss_14):.5f}; at 70M it gains {a7_gain_70:.3f} points for only ${a7_loss_70:+.5f}$ loss. Across $\kappa=0$ to 0.5, A4-OL1 gains {a4_span_14:.3f} and {a4_span_70:.3f} points of \Rmodel{{}} at 14M and 70M, while A7-OL1 gains {a7_span_14:.3f} and {a7_span_70:.3f} points. The result is descriptive one-seed evidence across two sizes; 410M remains unobserved.

{_family_table(data, "A4-OL1", "tab:a4-ol1-scale")}
\FloatBarrier

{_family_table(data, "A7-OL1", "tab:a7-ol1-scale")}
\FloatBarrier

The 70M post-hoc controls preserve the same local shape as 14M: low target sparsity first increases logical opportunity at small loss cost, followed by a steep quality penalty. TEAL clips \code{{a,m,h,z}} at evaluation only; it is not trained gating. Its source artifacts did not record downstream \code{{q,k,v}} activation mass, so those quantities are not inferred from \Rmodel{{}}.

{_teal_70_table(data)}
\FloatBarrier

\clearpage
\subsection{{Trained and post-hoc full-pass frontier}}

Figure~\ref{{fig:fullpass-frontier}} places both model sizes on one absolute validation-loss--\Rmodel{{}} plane. It includes all 20 trained A4-OL1/A7-OL1 endpoints and all 40 A0/A1-H post-hoc TEAL points. The vertical display ends at loss 6, but the full trajectories are retained and visibly exit the panel; every coordinate remains in Analysis 010's machine-readable reduction and complete Markdown tables.

\begin{{center}}
  \includegraphics[width=0.98\textwidth]{{../../../analyses/010-2026-09-01-pythia14m-vs-70m-selected-ladder/figures/01-pythia14m-vs-70m-selected-ladder.pdf}}
  \captionof{{figure}}{{Selected Pythia-14M and 70M trained and post-hoc frontier. Filled markers identify 14M and open markers 70M; magenta/black trajectories are A0/A1-H post-hoc TEAL and purple/sky-blue trajectories are trained A4-OL1/A7-OL1. Points above loss 6 remain in their plotted series and continue off-scale. Lines indicate dose or target order only. \Rmodel{{}} is logical zero-product opportunity, not measured speedup. Source: \href{{../../../analyses/010-2026-09-01-pythia14m-vs-70m-selected-ladder/observations/O001-pythia14m-vs-70m-selected-ladder.md}}{{Analysis 010, Observation O001}}.}}
  \label{{fig:fullpass-frontier}}
\end{{center}}
"""


def build_report() -> str:
    data = _read_json(ANALYSIS_DATA)
    billing = _read_json(BILLING_DATA)
    if data.get("schema_version") != 2 or data.get("status") != "complete_verified_analysis":
        raise ValueError("Analysis 010 reduction is not the verified schema")
    if len(data["trained_endpoints"]) != 20 or len(data["teal_points"]) != 40:
        raise ValueError("Analysis 010 endpoint coverage is incomplete")

    text = BASE_TEX.read_text(encoding="utf-8")
    for old, new, label in (
        (r"\title{Status Report Number 1}", r"\title{Status Report Number 2}", "title"),
        (r"\date{31 August 2026}", r"\date{1 September 2026}", "date"),
        (
            r"pdftitle={Status Report Number 1}",
            r"pdftitle={Status Report Number 2}",
            "PDF title",
        ),
        (
            r"\textbf{Paper-scale full pass.} RunPod A100 execution with 712 optimizer boundaries, global batch 1,024, and 1,493,172,224 training input tokens per condition. Conditions run concurrently across independent GPUs.",
            r"\textbf{Paper-scale full pass.} RunPod A100 execution at 14M and H200 execution at 70M, with 712 optimizer boundaries, global batch 1,024, and 1,493,172,224 training input tokens per condition. Conditions run concurrently across independent GPUs.",
            "paper-scale description",
        ),
        (r"A0 & 1 & Full pass complete: Run 004 & 0/1 & 0/1 \\", r"A0 & 1 & Full pass complete: Run 004 & 1/1 + TEAL: Run 018 & 0/1 \\", "A0 scope"),
        (r"A1-H & 1 & Full pass complete: Run 004 & 0/1 & 0/1 \\", r"A1-H & 1 & Full pass complete: Run 004 & 1/1 + TEAL: Run 018 & 0/1 \\", "A1-H scope"),
        (r"A4-OL1 & 5 & 5/5 full pass: Run 015 & 0/5 & 0/5 \\", r"A4-OL1 & 5 & 5/5 full pass: Run 015 & 5/5 full pass: Run 018 & 0/5 \\", "A4-OL1 scope"),
        (r"A7-OL1 & 5 & 5/5 full pass: Run 014 & 0/5 & 0/5 \\", r"A7-OL1 & 5 & 5/5 full pass: Run 014 & 5/5 full pass: Run 018 & 0/5 \\", "A7-OL1 scope"),
    ):
        text = _replace_once(text, old, new, label)

    compute_start = text.index(r"\section{RunPod compute}")
    results_start = text.index(r"\section{Results}")
    text = text[:compute_start] + _compute_section(billing) + text[results_start:]

    status_row = r"""7 & A7 $\rightarrow$ A7-OL1 & Five matched full-pass pairs in Analysis 008; descriptive \\
\bottomrule"""
    updated_status_row = r"""7 & A7 $\rightarrow$ A7-OL1 & Five matched full-pass pairs in Analysis 008; descriptive \\
Scale & 14M $\rightarrow$ selected 70M ladder & Four-family promotion complete in Analysis 010; descriptive \\
\bottomrule"""
    text = _replace_once(text, status_row, updated_status_row, "analysis-status scale row")

    frontier_start = text.index(r"\subsection{Trained and post-hoc full-pass frontier}")
    appendix_start = text.index(r"\appendix")
    text = text[:frontier_start] + _scale_section(data) + "\n\n" + text[appendix_start:]

    old_appendix_cost = r"Run 012 used five A100 80 GB Pods for 70--100 minutes per condition and cost \$14.77. Including this historical run, total recorded spend across the six main-ladder runs plus Run 012 is at least \$103.16; the Run 015 billing caveat stated in Section~3 still applies."
    historical = billing["historical_run_012"]["total_cost_usd"]
    combined = billing["total_including_historical_run_012_usd"]
    new_appendix_cost = rf"Run 012 used five A100 80 GB Pods for 70--100 minutes per condition and cost \${historical:.2f} in the current Pod-ID-reconciled billing refresh. Including this historical run, current Pod spend across all report rows is \${combined:.2f}; the independent retained network-volume charge remains excluded."
    text = _replace_once(text, old_appendix_cost, new_appendix_cost, "historical cost")
    return text


def main() -> None:
    content = build_report()
    temporary = OUTPUT_TEX.with_suffix(".tex.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(OUTPUT_TEX)
    print(f"wrote {OUTPUT_TEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
