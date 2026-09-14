"""Figure 07 v3: one native A0 latency reference for all full-model points."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_figure_v2", HERE / "07_kernel_realization_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
base = v2.base


def read_evidence():
    data = v2.read_evidence()
    rows = json.loads(base.INVESTIGATION.read_text(encoding="utf-8"))
    a0, = [row for row in rows if row["recipe"] == "A0"]
    assert a0["condition"] == "c01"
    reference = a0["native_baseline_gm_ms"]
    assert reference == a0["full_native_gm_ms"] and reference > 0
    by_condition = {row["condition"]: row for row in rows}
    data["common_a0_reference"] = {
        "condition": a0["condition"], "recipe": "A0",
        "checkpoint_evidence_id": a0["checkpoint_evidence_id"],
        "native_gm_ms": reference,
        "source_field": "native_baseline_gm_ms (full-K050 process native controls)",
        "aggregation": "Geometric mean of 64 inputs x 7 passes x 3 processes (1,344 host latencies)",
    }
    points = []
    for point in data["points"]:
        if point["candidate"] != "k050":
            continue
        row = by_condition[point["condition"]]
        assert row["checkpoint_evidence_id"] == point["evidence_id"]
        assert math.isclose(row["s_model_percent"], point["sparsity_percent"], abs_tol=1e-12)
        points.append({key: point[key] for key in
                       ("condition", "evidence_id", "family", "visual_family", "dose", "sparsity_percent")})
        points[-1].update({
            "candidate": "k050", "full_candidate_gm_ms": row["full_candidate_gm_ms"],
            "checkpoint_native_gm_ms": row["native_baseline_gm_ms"],
            "checkpoint_native_relative_speedup": row["full_speedup"],
            "common_a0_native_gm_ms": reference,
            "a0_relative_speedup": reference / row["full_candidate_gm_ms"],
        })
    assert len(points) == 30
    data["common_a0_points"] = points
    data["common_a0_regression"] = base.regression(points, "sparsity_percent", "a0_relative_speedup")
    data["question"] = "How does model-wide sparsity relate to full-model speed relative to one native A0 reference, and to matched projection-skipping gain?"
    data["variant"] = "v3; panel (a) uses common native A0 latency / each checkpoint's full K050 latency; panel (b) is unchanged from v2"
    data["protocol"]["baseline"] = "Panel (a): one fixed native A0 geometric-mean latency from c01 full-K050 processes"
    data["retained_points_metric"] = "points and regression retain the original same-checkpoint native ratios; panel (a) displays common_a0_points and common_a0_regression"
    data["timing_metric"] = "Panel (a): GM native A0 latency / GM full K050 checkpoint latency; panel (b): GM all-skips-off / GM projection-on latency at the same checkpoint"
    data["interpretation"] = "Common-reference comparison across different trained models, not equal-quality acceleration or an isolated sparse-skipping effect. Retained cross-process timings are not newly interleaved A0/checkpoint pairs."
    data["v2_figure_sha256"] = hashlib.sha256((HERE / "figures/07-kernel-realization-v2.pdf").read_bytes()).hexdigest()
    return data


def make_figure(data):
    # Adapt only the plotting view; preserve the original native ratios in evidence.
    view = {**data, "regression": data["common_a0_regression"],
            "points": [{**p, "speedup": p["a0_relative_speedup"]} for p in data["common_a0_points"]]}
    fig = v2.make_figure(view)
    ax = fig.axes[0]
    ax.set(ylim=(.98, 1.50), ylabel="Full-model speedup vs. A0 (×)")
    ax.set_yticks([1., 1.1, 1.2, 1.3, 1.4, 1.5])
    for text in ax.texts:
        if text.get_text() == "1× native":
            text.set_text("1× native A0")
            text.set_position((29, 1.008))
        elif text.get_text().startswith("7-site + OL1"):
            text.set_position((28.5, 1.25))
    return fig


def main():
    data = read_evidence()
    (HERE / "data/kernel-realization-v3.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    fig.savefig(HERE / "figures/07-kernel-realization-v3.pdf",
                metadata={"Title": "K050 speedup against common native A0 and matched projection-skipping gain",
                          "Creator": "Analysis 021 / 07_kernel_realization_v3.py", "CreationDate": None})
    base.plt.close(fig)
    assert hashlib.sha256((HERE / "figures/07-kernel-realization-v2.pdf").read_bytes()).hexdigest() == data["v2_figure_sha256"]
    print(f'30 checkpoints; common native A0 = {data["common_a0_reference"]["native_gm_ms"]:.9f} ms; R² = {data["common_a0_regression"]["r2"]:.6f}')
    for point in data["common_a0_points"]:
        if point["condition"] in ("c20", "c30"):
            print(f'{point["condition"]} {point["family"]}: {point["a0_relative_speedup"]:.6f}×')


if __name__ == "__main__":
    main()
