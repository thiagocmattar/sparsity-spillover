"""Matched retained A7 endpoints: no pressure, h-only OL1, seven-site OL1."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "none": ROOT / "runs/013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json",
    "h": ROOT / "runs/032-2026-09-16-pythia14m-a7-h-only-ol1/artifacts/verification.json",
    "all7": ROOT / "runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/artifacts/verification.json",
}
KAPPAS = (0.0, 0.01, 0.05, 0.1, 0.5)


def main():
    evidence = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in SOURCES.items()}
    rows, lookup, identities = [], {}, set()
    for pressure, document in evidence.items():
        assert document["status"] == "verified"
        seen = set()
        for result in document["conditions"]:
            condition = result["condition"]
            kappa = float(condition["gate_threshold"])
            assert kappa not in seen and kappa in KAPPAS
            seen.add(kappa)
            assert condition["topology_id"] == "A7-Z-POST"
            assert set(condition["active_sites"]) == {"a", "m", "h", "z", "q_post", "k_post", "v"}
            expected = [] if pressure == "none" else ["h"] if pressure == "h" else condition["active_sites"]
            assert set(condition["pressure_sites"]) == set(expected)
            assert result["completed_steps"] == 712 and result["input_tokens"] == 1493172224
            identities.add((result["initial_parameter_sha256"], result["training_schedule_sha256"]))
            row = {"kappa": kappa, "pressure": pressure,
                   "validation_loss": result["final_validation_loss"],
                   "train_loss": result["final_train_loss"],
                   "R_model": result["R_model"], "R_block": result["R_block"],
                   "R_model_max": result["R_model_max"], "attempt_id": result["attempt_id"]}
            geometry = result.get("ol1", {})
            row.update({"pressure_capture_tensor_count": geometry.get("pressure_capture_tensor_count"),
                        "trust_saturated_boundaries": geometry.get("trust_saturated_boundary_count"),
                        "maximum_pressure_task_step_ratio": geometry.get("maximum_final_ratio")})
            row.update({"exact_zero_" + site: value for site, value in sorted(result["selected_site_exact_zero_fractions"].items())})
            rows.append(row)
            lookup[pressure, kappa] = row
        assert seen == set(KAPPAS)
    assert len(identities) == 1
    comparisons = []
    for kappa in KAPPAS:
        n, h, a = (lookup[p, kappa] for p in ("none", "h", "all7"))
        full_gain = a["R_model"] - n["R_model"]
        comparisons.append({
            "kappa": kappa,
            "h_minus_none_R_model_pp": 100 * (h["R_model"] - n["R_model"]),
            "all7_minus_none_R_model_pp": 100 * full_gain,
            "h_share_of_positive_all7_R_model_increment": (h["R_model"] - n["R_model"]) / full_gain if full_gain > 0 else None,
            "h_minus_none_validation_loss": h["validation_loss"] - n["validation_loss"],
            "h_minus_all7_validation_loss": h["validation_loss"] - a["validation_loss"],
        })
    output = HERE / "artifacts"
    output.mkdir(exist_ok=True)
    rows.sort(key=lambda row: (row["kappa"], ("none", "h", "all7").index(row["pressure"])))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"sources": {key: {"path": str(path.relative_to(ROOT)).replace('\\', '/'), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in SOURCES.items()},
               "coverage": "One matched seed, five thresholds, 712 updates per condition; final-checkpoint full validation (500 documents, 338 complete blocks, excluded tail 1444 tokens).",
               "metric_units": {"validation_loss": "mean next-token cross-entropy, nats", "R_model": "fraction of full-model dense scalar products logically avoidable", "exact_zero": "pooled integer zero count / pooled integer element count"},
               "rows": rows, "matched_differences": comparisons}
    (output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["| κ | Validation loss: none / h / all7 | R_model (%): none / h / all7 |", "|---:|---:|---:|"]
    for kappa in KAPPAS:
        group = [lookup[p, kappa] for p in ("none", "h", "all7")]
        losses = " / ".join(f"{r['validation_loss']:.4f}" for r in group)
        opportunities = " / ".join(f"{100*r['R_model']:.2f}" for r in group)
        lines.append(f"| {kappa:g} | {losses} | {opportunities} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(comparisons[-1], indent=2))


if __name__ == "__main__":
    main()
