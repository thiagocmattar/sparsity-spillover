"""Clean 14M overview with the audited A4 and recovered A7 h-only OL1 curves."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OVERVIEW = ROOT / "analyses/021-2026-09-10-training-results-figures/data/14m-quality-sparsity.json"
A4_AUDIT = ROOT / "analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/figure_data.json"
A7_VERIFICATION = HERE / "artifacts/verification.json"
KAPPAS = [0.0, 0.01, 0.05, 0.1, 0.5]
YLIM = (5.03, 6.23)
COLORS = {"baseline": "#60656C", "relu": "#22836D", "A4": "#2878B5", "A7": "#C96024"}


def read_evidence():
    sources = {}

    def source(path, expected_sha256=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None:
            assert digest == expected_sha256, f"Source hash changed: {path}"
        sources[path.relative_to(ROOT).as_posix()] = digest
        return path

    overview = json.loads(source(OVERVIEW).read_text(encoding="utf-8"))
    for path, digest in overview["sources"].items():
        source(ROOT / path, digest)
    a4 = json.loads(source(A4_AUDIT).read_text(encoding="utf-8"))
    a7 = json.loads(source(A7_VERIFICATION).read_text(encoding="utf-8"))
    assert a4["status"] == "complete_verified_analysis"
    assert a7["status"] == "verified" and a7["condition_count"] == 5
    identity = a4["matched_identity"]
    for field in ("initial_parameter_sha256", "training_schedule_sha256"):
        assert identity[field] == a7[field]
    for field in ("documents", "sequences", "input_tokens", "excluded_tail_tokens", "seed_count"):
        assert a4["coverage"][field] == overview["coverage"][field]
    audit = a4["realization_audit"]["run012_h_only"]
    assert audit["realized_pressure_sites"] == ["h"]
    for field in ("inherited_source", "wrapper"):
        source(ROOT / audit[field], audit[field + "_sha256"])
    verification = a4["source_verification"]["run012_h_only"]
    source(ROOT / verification["path"], verification["sha256"])

    # Preserve every point from the reference overview, including clipped points
    # outside its display range. Only the two requested series are added.
    rows = overview["points"]
    assert len(rows) == 46
    for row in a4["series"]:
        if row["series_id"] != "run012_h_only":
            continue
        assert row["realized_pressure_sites"] == ["h"]
        for field in ("activation_statistics", "logical_products", "manifest"):
            source(ROOT / row["source_files"][field], row["source_files"][field + "_sha256"])
        counts = row["logical_product_counts"]
        rows.append({
            "id": f"14M:A4-OL1-H:{row['kappa']}", "kind": "trained",
            "family": "A4-OL1-H", "dose": row["kappa"],
            "loss": row["final_validation_loss"], "R_model": row["R_model"],
            "source": row["source_files"]["logical_products"],
            "attempt_id": row["attempt_id"], "pressure_sites": ["h"],
            "counts": {"block_zero_product_count": counts["zero_product_count"],
                       "model_product_count": counts["model_product_count"]},
            "coverage": a4["coverage"],
        })
    for row in a7["conditions"]:
        condition = row["condition"]
        assert condition["pressure_sites"] == ["h"]
        assert condition["pressure_method"] == "orthogonal_l1"
        assert condition["pressure_weight"] == condition["step_budget"] == 1
        assert set(condition["one_sided_sites"]) == {"a", "m", "h", "z"}
        assert set(condition["symmetric_sites"]) == {"q_post", "k_post", "v"}
        assert row["completed_steps"] == identity["optimizer_steps_per_condition"]
        assert row["input_tokens"] == identity["training_input_tokens_per_condition"]
        assert row["ol1"]["pressure_capture_tensor_count"] == 6
        for field in ("initial_parameter_sha256", "training_schedule_sha256"):
            assert row[field] == identity[field]
        path = HERE / "artifacts/attempts" / row["attempt_id"] / "diagnostics/logical_products.json"
        logical = json.loads(source(path).read_text(encoding="utf-8"))
        coverage = logical["coverage"]
        assert coverage["complete_block_coverage"]
        for field in ("sequences", "input_tokens", "excluded_tail_tokens"):
            assert coverage[field] == overview["coverage"][field]
        measured = logical["measured"]
        rows.append({
            "id": f"14M:A7-OL1-H:{condition['gate_threshold']}", "kind": "trained",
            "family": "A7-OL1-H", "dose": condition["gate_threshold"],
            "loss": row["final_validation_loss"], "R_model": row["R_model"],
            "source": path.relative_to(ROOT).as_posix(),
            "attempt_id": row["attempt_id"], "pressure_sites": ["h"],
            "counts": {key: measured[key] for key in ("block_zero_product_count", "model_product_count")},
            "coverage": overview["coverage"],
        })
    for family in ("A4-OL1-H", "A7-OL1-H"):
        assert sorted(row["dose"] for row in rows if row["family"] == family) == KAPPAS
    assert len(rows) == 56
    for row in rows:
        counts = row["counts"]
        assert all(isinstance(value, int) for value in counts.values())
        ratio = counts["block_zero_product_count"] / counts["model_product_count"]
        assert math.isclose(ratio, row["R_model"], rel_tol=0, abs_tol=1e-12)
        assert math.isfinite(row["loss"])
    return {
        "sources": sources, "coverage": overview["coverage"],
        "matched_h_only_identity": identity,
        "a4_pressure_realization_audit": audit,
        "x_definition": "100 * pooled block_zero_product_count / model_product_count; logical opportunity, not speedup",
        "y_definition": "ordinary final-checkpoint validation loss; not diagnostic-pass loss",
        "display_loss_limits": list(YLIM),
        "display_note": "Reference overview plus A4/A7 h-only OL1; no point annotations, callouts, or ceiling guides. Reference axis limits retained.",
        "points": rows,
    }


def make_figure(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.30, top=0.985)
    ax.set(xlim=(-0.65, 31.3), ylim=YLIM,
           xlabel=r"Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)",
           ylabel="Validation loss (lower is better)")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([5.2, 5.4, 5.6, 5.8, 6.0, 6.2])
    ax.grid(axis="y", color="#E8E9EC", linewidth=0.55)
    ax.set_axisbelow(True)
    handles = {}

    def plot(family, label, color, marker, linestyle, filled=True, kind="trained", emphasis=False):
        rows = sorted((r for r in data["points"] if r["family"] == family and r["kind"] == kind),
                      key=lambda r: -1 if r["dose"] is None else r["dose"])
        x = [100 * r["counts"]["block_zero_product_count"] / r["counts"]["model_product_count"] for r in rows]
        y = [r["loss"] for r in rows]
        line, = ax.plot(x, y, label=label, color=color, marker=marker,
                        markersize=5 if emphasis else (3.3 if kind == "clipped" else 4.5),
                        markerfacecolor=color if filled else "white", markeredgecolor=color,
                        markeredgewidth=0.8, linestyle=linestyle,
                        linewidth=1.3 if emphasis else 0.85,
                        zorder=4 if emphasis else (2 if kind == "clipped" else 3),
                        gid=f"{kind}:{family}")
        handles[(family, kind)] = line

    for family, color in [("A0", COLORS["baseline"]), ("A1-H", COLORS["relu"])]:
        plot(family, ("Baseline" if family == "A0" else "ReLU") + " + post-hoc clipping",
             color, "o", ":", filled=False, kind="clipped")
    plot("A0", "Baseline", COLORS["baseline"], "o", "None")
    plot("A1-H", "ReLU", COLORS["relu"], "s", "None")
    plot("A1-H-OL1", "ReLU + OL1", COLORS["relu"], "^", "--")
    for family, marker, h_marker, count in [("A4", "D", "s", 4), ("A7", "^", "o", 7)]:
        color = COLORS[family]
        plot(family, f"{family}, no OL1", color, marker, "-", filled=False)
        plot(family + "-OL1", f"{family} + OL1 (all {count} sites)", color, marker, "--")
        plot(family + "-OL1-H", f"{family} + OL1 (h only)", color, h_marker, "-.", emphasis=True)
    order = [
        ("A4", "trained"), ("A4-OL1-H", "trained"), ("A4-OL1", "trained"), ("A0", "trained"),
        ("A7", "trained"), ("A7-OL1-H", "trained"), ("A7-OL1", "trained"), ("A1-H", "trained"),
        ("A1-H-OL1", "trained"), ("A0", "clipped"), ("A1-H", "clipped"),
    ]
    fig.legend(handles=[handles[key] for key in order], ncol=3, loc="lower center",
               bbox_to_anchor=(0.54, 0.01), frameon=False, fontsize=7.2,
               handlelength=2.7, columnspacing=1.8, handletextpad=0.6, labelspacing=0.65)
    assert not ax.texts, "The requested figure must have no point labels or callouts"
    assert sum(len(line.get_xdata()) for line in ax.lines) == 56
    return fig


def main():
    data = read_evidence()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    data_path = HERE / "data/14m-quality-sparsity-h-only.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    output = HERE / "figures/01-14m-quality-sparsity-h-only.pdf"
    fig.savefig(output, metadata={"Title": "Pythia-14M quality and sparsity with h-only OL1",
                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Verified 46 reference points plus 10 h-only points; wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
