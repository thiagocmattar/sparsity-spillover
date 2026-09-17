"""Plot all retained 70M trained endpoints with one final-loss convention."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = (
    "018-2026-09-01-pythia70m-selected-ladder-canonical-init",
    "034-2026-09-17-pythia70m-h-only-ol1",
)
KAPPAS = [0.0, 0.01, 0.05, 0.1, 0.5]
STYLES = {
    "Baseline (GeLU)": ("#555B63", "D", "", True),
    "ReLU": ("#19856B", "s", "", True),
    "A4 + OL1(all)": ("#2878B5", "o", "-", True),
    "A4 + OL1(h)": ("#2878B5", "o", "--", False),
    "A7 + OL1(all)": ("#C45A25", "^", "-", True),
    "A7 + OL1(h)": ("#C45A25", "^", "--", False),
}
COVERAGE = {
    "sequences": 338,
    "input_tokens": 692_224,
    "source_tokens": 693_668,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}


def read_evidence():
    sources, rows, identities = {}, [], set()

    def read(path):
        raw = path.read_bytes()
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for run_name in RUNS:
        run = ROOT / "runs" / run_name
        verified = read(run / "artifacts/verification.json")
        assert verified["status"] == "verified" and verified["evidence_label"] == "valid"
        assert len(verified["conditions"]) == verified["condition_count"]
        for result in verified["conditions"]:
            attempt = run / "artifacts/attempts" / result["attempt_id"]
            manifest = read(attempt / "manifest.json")
            metrics = read(attempt / "metrics.json")
            logical = read(attempt / "diagnostics/logical_products.json")
            condition = result["condition"]
            assert manifest["condition"] == metrics["condition"] == condition
            assert manifest["status"] == "completed" and result["status"] == "verified"
            assert manifest["completed_steps"] == 712
            assert manifest["input_tokens"] == 1_493_172_224
            assert not manifest["gradient_overflow_steps"]
            identities.add((manifest["initial_parameter_sha256"], manifest["training_schedule_hash"]))
            final = metrics["validation"]["final"]
            for measurement in (final, logical["coverage"]):
                assert all(measurement[k] == v for k, v in COVERAGE.items())
            measured = logical["measured"]
            numerator = measured["block_zero_product_count"]
            denominator = measured["model_product_count"]
            assert isinstance(numerator, int) and isinstance(denominator, int)
            assert sum(x["zero_product_count"] for x in measured["per_operation"].values()) == numerator
            assert measured["block_product_count"] + measured["lm_head_product_count"] == denominator
            fraction = numerator / denominator
            assert math.isclose(fraction, measured["R_model"], abs_tol=1e-15, rel_tol=0)
            assert math.isclose(fraction, result["R_model"], abs_tol=1e-15, rel_tol=0)
            assert final["loss"] == result["final_validation_loss"]
            assert math.isfinite(final["loss"])
            if condition["id"] == "a0-gelu":
                family = "Baseline (GeLU)"
            elif condition["id"] == "a1h-relu":
                family = "ReLU"
            else:
                scope = "h" if condition["pressure_sites"] == ["h"] else "all"
                assert condition["pressure_method"] == "orthogonal_l1"
                assert condition["pressure_weight"] == condition["step_budget"] == 1.0
                if scope == "all":
                    assert set(condition["pressure_sites"]) == set(condition["active_sites"])
                topology = "A4" if condition["topology_id"] == "A4-Z" else "A7"
                assert condition["topology_id"] in {"A4-Z", "A7-Z-POST"}
                family = f"{topology} + OL1({scope})"
            rows.append({
                "family": family,
                "condition_id": condition["id"],
                "condition": condition,
                "kappa": condition["gate_threshold"],
                "run": run_name,
                "attempt_id": result["attempt_id"],
                "checkpoint_content_sha256": result["checkpoint_content_sha256"],
                "final_validation_loss": final["loss"],
                "logical_pass_loss_for_audit_only": logical["coverage"]["loss"],
                "block_zero_product_count": numerator,
                "model_product_count": denominator,
                "S_model_percent": 100 * fraction,
                "R_model": fraction,
                "training_devices": manifest["environment"]["devices"],
            })
    assert len(rows) == 22 and len(identities) == 1
    assert len({(r["run"], r["condition_id"]) for r in rows}) == 22
    for family in STYLES:
        selected = [r for r in rows if r["family"] == family]
        if family.startswith(("A4", "A7")):
            assert sorted(r["kappa"] for r in selected) == KAPPAS
        else:
            assert len(selected) == 1
    initial, schedule = identities.pop()
    return {
        "status": "verified_source_reduction",
        "scope": "All 22 completed 70M trained endpoints as of 2026-09-17; no post-hoc clipping points.",
        "loss_source": "metrics.validation.final.loss, consistently for all 22 final checkpoints",
        "sparsity_definition": "100 * pooled block_zero_product_count / model_product_count; logical opportunity, not runtime speedup",
        "coverage": {"validation_documents": 500, **COVERAGE},
        "initial_parameter_sha256": initial,
        "training_schedule_sha256": schedule,
        "source_sha256": sources,
        "maximum_absolute_final_vs_logical_pass_loss_difference": max(
            abs(r["final_validation_loss"] - r["logical_pass_loss_for_audit_only"]) for r in rows
        ),
        "points": rows,
    }


def make_figure(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "pdf.fonttype": 42,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.245, top=0.965)
    for family, (color, marker, linestyle, filled) in STYLES.items():
        rows = sorted((r for r in data["points"] if r["family"] == family),
                      key=lambda r: -1 if r["kappa"] is None else r["kappa"])
        ax.plot([r["S_model_percent"] for r in rows],
                [r["final_validation_loss"] for r in rows],
                label=family, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.25, markersize=6, markeredgewidth=1.05,
                markerfacecolor=color if filled else "white", zorder=3,
                gid=family)
    ax.set(xlim=(-0.8, 43), ylim=(4.0, 5.5),
           xlabel=r"Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)",
           ylabel="Validation loss (nats)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(axis="y", color="#E7E9ED", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.54, 0.025), frameon=False, fontsize=8.6,
               handlelength=2.3, columnspacing=1.6, labelspacing=0.9)
    assert sum(len(line.get_xdata()) for line in ax.lines) == 22
    assert all(ax.get_xlim()[0] <= r["S_model_percent"] <= ax.get_xlim()[1]
               and ax.get_ylim()[0] <= r["final_validation_loss"] <= ax.get_ylim()[1]
               for r in data["points"])
    return fig


def main():
    data = read_evidence()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "data/70m-quality-sparsity.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    output = HERE / "figures/01-70m-quality-sparsity.pdf"
    fig.savefig(output, metadata={"Title": "Pythia-70M: all trained quality-sparsity endpoints",
                                 "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(json.dumps({"figure": output.relative_to(ROOT).as_posix(),
                      "points": len(data["points"]), "verified_source_files": len(data["source_sha256"]),
                      "max_loss_pass_difference": data["maximum_absolute_final_vs_logical_pass_loss_difference"]}))


if __name__ == "__main__":
    main()
