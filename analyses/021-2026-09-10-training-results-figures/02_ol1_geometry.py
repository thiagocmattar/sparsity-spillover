"""OL1 geometry from every retained 14M multisite optimizer-boundary log."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
FAMILIES = ("A4-OL1", "A7-OL1")
THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.5)
STEPS = 712
CODE_NAMES = {"pressure.py", "optimization.py", "optimizer_boundary.py", "training.py"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_step(row, pressure):
    """Check the logged, global adaptive geometry; never substitute ideal zeros."""
    eps, weight, budget = (pressure[k] for k in ("eps", "weight", "step_budget"))
    u_norm = row["task_direction_norm"]
    w_norm = row["pressure_direction_norm_raw"]
    ratio = row["pressure_to_task_ratio_raw"]
    safe_norm = ratio * (u_norm + eps) / weight
    assert all(math.isfinite(v) and v > 0 for v in (u_norm, w_norm, ratio, safe_norm))
    assert row["projection_applied"] == (
        row["task_pressure_dot_before"] < 0 and u_norm**2 > eps
    )
    for field, expected in (
        ("task_pressure_cosine_before", row["task_pressure_dot_before"] / (u_norm * w_norm + eps)),
        ("task_pressure_cosine_after", row["task_pressure_dot_after"] / (u_norm * safe_norm + eps)),
        ("trust_scale", min(1.0, budget / (ratio + eps))),
        ("pressure_to_task_ratio_final", ratio * row["trust_scale"]),
    ):
        assert math.isclose(row[field], expected, rel_tol=1e-12, abs_tol=1e-16), field
    assert row["pressure_weight"] == weight
    assert row["ol1_correction_applied"]
    assert not row["optimizer_step_skipped"] and not row["gradient_overflow"]


def read_evidence():
    selected = [r for r in json.loads(COHORT.read_text(encoding="utf-8"))["trained"]
                if r["scale"] == "14M" and r["family"] in FAMILIES]
    assert len(selected) == 10
    assert {(r["family"], r["dose"]) for r in selected} == {
        (family, dose) for family in FAMILIES for dose in THRESHOLDS
    }
    sources = {COHORT.relative_to(ROOT).as_posix(): sha256(COHORT)}
    code_sources = {}
    conditions = []
    for item in sorted(selected, key=lambda r: (r["family"], r["dose"])):
        attempt = ROOT / item["source"]
        manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        pressure = manifest["activation_pressure"]
        assert manifest["status"] == "completed" and manifest["completed_steps"] == STEPS
        assert pressure == item["identity"]["pressure"]
        assert pressure["method"] == "orthogonal_l1"
        assert pressure["weight"] == pressure["step_budget"] == 1.0
        assert pressure["eps"] == 1e-12
        sites = ["a", "m", "h", "z"] if item["family"] == "A4-OL1" else [
            "a", "m", "h", "q_post", "k_post", "v", "z"
        ]
        assert pressure["sites"] == sites
        for name in ("manifest.json", "events.jsonl"):
            path = attempt / name
            sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        # Match the geometry and optimizer ordering to the executed code identity.
        for entry in manifest["run_code"]["files"]:
            if Path(entry["path"]).name not in CODE_NAMES:
                continue
            path = (attempt.parents[2] / entry["path"]).resolve()
            contents = path.read_bytes()
            canonical = "canonical_lf_bytes" in entry
            if canonical:
                contents = contents.replace(b"\r\n", b"\n")
            assert hashlib.sha256(contents).hexdigest() == entry["sha256"], path
            code_sources[path.relative_to(ROOT).as_posix()] = {
                "sha256": entry["sha256"], "canonical_lf": canonical,
            }
        events = [json.loads(line) for line in (attempt / "events.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in events if r["event"] == "train"]
        assert [r["step"] for r in rows] == list(range(1, STEPS + 1))
        assert len({r["condition_id"] for r in rows}) == 1
        for row in rows:
            audit_step(row, pressure)
            assert row["eligible_parameter_tensors"] == 69
            assert row["skipped_parameter_tensors"] == 7
            assert row["pressure_capture_tensor_count"] == 6 * len(sites)
        conditions.append({
            "id": item["id"], "family": item["family"], "kappa": item["dose"],
            "source": item["source"], "pressure": pressure, "rows": rows,
        })
    return conditions, {"source_sha256": sources, "verified_historical_code": code_sources}


def summarize(rows):
    conflicts = [r for r in rows if r["task_pressure_dot_before"] < 0]
    caps = sum(r["trust_scale"] < 1 for r in rows)
    return {
        "observations": len(rows), "conflict_steps": len(conflicts),
        "conflict_percent": 100 * len(conflicts) / len(rows),
        "median_conflicting_pre_cosine": float(np.median([
            r["task_pressure_cosine_before"] for r in conflicts])),
        "max_abs_conflicting_post_cosine": max(abs(r["task_pressure_cosine_after"]) for r in conflicts),
        "cap_active_steps": caps, "cap_active_percent": 100 * caps / len(rows),
        "median_pre_cap_ratio": float(np.median([r["pressure_to_task_ratio_raw"] for r in rows])),
        "zero_learning_rate_steps": sum(r["learning_rate"] == 0 for r in rows),
    }


def summary_document(conditions, provenance):
    return {
        "scope": "Pythia-14M A4-OL1/A7-OL1; all five thresholds; 712 optimizer steps each",
        "unit": "one optimizer step x one training condition; global eligible-parameter geometry",
        "lambda": 1, "budget": 1, "eps": 1e-12,
        "conflict_rule": "task_pressure_dot_before < 0",
        "cap_active_rule": "trust_scale < 1",
        "cosines": "logged stabilized global adaptive-direction cosines, unchanged",
        "median_trace": "arithmetic median of r/b across all ten conditions at each step; no smoothing",
        **provenance,
        "pooled": summarize([r for c in conditions for r in c["rows"]]),
        "by_family": {family: summarize([r for c in conditions if c["family"] == family
                                         for r in c["rows"]]) for family in FAMILIES},
        "conditions": [{k: v for k, v in c.items() if k != "rows"} | summarize(c["rows"])
                       for c in conditions],
    }


def make_figure(conditions, summary):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.9, 2.85))
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.18, top=0.76, wspace=0.45)
    fig.suptitle("OL1 task–pressure geometry and budget saturation", fontsize=9.2, y=0.965)
    left.set_title("(a) Conflict-conditioned projection", fontsize=7.6, pad=10, loc="left")
    right.set_title(r"(b) Norm-budget saturation at $\lambda = 1$", fontsize=7.6, pad=10, loc="left")
    rows = [r for c in conditions for r in c["rows"]]
    left.axvline(0, color="#B1B6BC", linewidth=0.6, zorder=0)
    left.axhline(0, color="#B1B6BC", linewidth=0.6, zorder=0)
    left.plot([-.15, .015], [-.15, .015], color="#D1D5D9", linewidth=0.6, zorder=0)
    left.scatter([r["task_pressure_cosine_before"] for r in rows],
                 [r["task_pressure_cosine_after"] for r in rows],
                 s=5, alpha=0.15, color="#337887", edgecolors="none", zorder=2)
    left.set(xlim=(-.15, .015), ylim=(-.15, .015),
             xlabel=r"Pre-projection cosine $\cos(u,w)$",
             ylabel=r"Post-projection cosine $\cos(u,\widetilde{w})$")
    left.set_xticks([-.15, -.10, -.05, 0])
    left.set_yticks([-.15, -.10, -.05, 0])
    pooled = summary["pooled"]
    left.text(.045, .42, f"Conflict: {pooled['conflict_percent']:.1f}% of steps",
              transform=left.transAxes, fontsize=7.5,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    left.text(.045, .26, "Median pre-proj. cosine\nwhen conflicting: "
              + f"{pooled['median_conflicting_pre_cosine']:.3f}".replace("-", "−"),
              transform=left.transAxes, fontsize=7.2, linespacing=1.4,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    ratios = []
    styles = {"A4-OL1": "-", "A7-OL1": (0, (3, 2))}
    colors = {"A4-OL1": "#7E929E", "A7-OL1": "#8B827C"}
    for condition in conditions:
        values = [r["pressure_to_task_ratio_raw"] / condition["pressure"]["step_budget"]
                  for r in condition["rows"]]
        ratios.append(values)
        right.plot(range(1, STEPS + 1), values, color=colors[condition["family"]],
                   linestyle=styles[condition["family"]], alpha=.52, linewidth=.55)
    right.plot(range(1, STEPS + 1), np.median(ratios, axis=0), color="#222B32", linewidth=1.2)
    right.axhline(1, color="#555D63", linewidth=.65, linestyle=(0, (1.5, 2.5)), zorder=0)
    right.set(yscale="log", xlim=(1, STEPS), ylim=(.055, 6500),
              xlabel="Optimizer step", ylabel=r"Pre-cap norm ratio $r/b$")
    right.set_xticks([1, 200, 400, 600])
    right.set_yticks([.1, 1, 10, 100, 1000])
    right.minorticks_off()
    right.text(.98, .965, f"Cap active: {pooled['cap_active_percent']:.1f}% of steps",
               transform=right.transAxes, ha="right", va="top", fontsize=7.5)
    split = summary["by_family"]
    right.text(.98, .86,
               f"4-site: {split['A4-OL1']['cap_active_percent']:.1f}%  |  "
               f"7-site: {split['A7-OL1']['cap_active_percent']:.1f}%",
               transform=right.transAxes, ha="right", va="top", fontsize=6.5)
    handles = [Line2D([], [], color=colors[f], linestyle=styles[f], linewidth=.9,
                       label=f"{n}-site") for f, n in zip(FAMILIES, (4, 7))]
    handles.append(Line2D([], [], color="#222B32", linewidth=1.2, label="Median"))
    right.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
                 bbox_to_anchor=(.5, -.30), fontsize=6.8, handlelength=1.5,
                 handletextpad=.4, columnspacing=.8, borderaxespad=0)
    return fig


def main():
    conditions, provenance = read_evidence()
    summary = summary_document(conditions, provenance)
    (HERE / "data/14m-ol1-geometry.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    figure = make_figure(conditions, summary)
    figure.savefig(HERE / "figures/02-14m-ol1-geometry.pdf", metadata={
        "Title": "OL1 task-pressure geometry and budget saturation", "CreationDate": None,
        "ModDate": None,
    })
    plt.close(figure)
    print(json.dumps(summary["pooled"], indent=2))


if __name__ == "__main__":
    main()
