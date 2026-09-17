"""Archive the approved 30 endpoints and audit both final-checkpoint loss passes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = ROOT / "runs/032-2026-09-16-pythia14m-a7-h-only-ol1/data/14m-quality-sparsity-h-only.json"
FIGURE = SNAPSHOT.parent.parent / "figures/01-14m-quality-sparsity-h-only.pdf"
A4_AUDIT = ROOT / "analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/figure_data.json"
FAMILIES = {"A4": "A4", "A4-OL1": "A4+OL1(all)", "A4-OL1-H": "A4+OL1(h)",
            "A7": "A7", "A7-OL1": "A7+OL1(all)", "A7-OL1-H": "A7+OL1(h)"}
KAPPAS = [0.0, 0.01, 0.05, 0.1, 0.5]


def build():
    sources = {}

    def record(path, expected=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected is None or digest == expected, f"Changed source: {path}"
        sources[path.relative_to(ROOT).as_posix()] = digest
        return path

    def read(path, expected=None):
        return json.loads(record(path, expected).read_text(encoding="utf-8"))

    frozen = read(SNAPSHOT)
    record(FIGURE)
    for path, digest in frozen["sources"].items():
        record(ROOT / path, digest)
    audit = read(A4_AUDIT)
    assert audit["status"] == "complete_verified_analysis"
    realization = audit["realization_audit"]["run012_h_only"]
    assert realization["realized_pressure_sites"] == ["h"]
    rows = []
    identities = set()
    for original in frozen["points"]:
        family = original["family"]
        if original["kind"] != "trained" or family not in FAMILIES:
            continue
        path = ROOT / original["source"]
        attempt = path.parents[1] if path.name == "logical_products.json" else path
        run = attempt.parents[2]
        manifest = read(attempt / "manifest.json")
        metrics = read(attempt / "metrics.json")
        logical = read(attempt / "diagnostics/logical_products.json")
        verification = read(run / "artifacts/verification.json")
        assert verification["status"] == "verified"
        verified = next(r for r in verification["conditions"] if r["attempt_id"] == attempt.name)
        condition = manifest["condition"]
        assert manifest["status"] == "completed"
        assert condition["gate_threshold"] == original["dose"]
        assert manifest["completed_steps"] == 712 and manifest["input_tokens"] == 1493172224
        assert manifest["seeds"] == {"model": 1234, "data_order": 1234}
        identities.add((manifest["initial_parameter_sha256"], manifest["training_schedule_hash"]))
        ordinary = metrics["validation"]["final"]
        eager = metrics["validation"]["logical_product_diagnostic_eager"]
        for coverage in (ordinary, eager, logical["coverage"]):
            assert coverage["complete_block_coverage"]
            assert (coverage["sequences"], coverage["input_tokens"], coverage["excluded_tail_tokens"]) == (338, 692224, 1444)
        assert ordinary["loss"] == verified["final_validation_loss"]
        assert eager["loss"] == logical["coverage"]["loss"]
        counts = {k: logical["measured"][k] for k in ("block_zero_product_count", "model_product_count")}
        assert counts == original["counts"] and all(isinstance(v, int) for v in counts.values())
        ratio = counts["block_zero_product_count"] / counts["model_product_count"]
        assert math.isclose(ratio, original["R_model"], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(ratio, verified["R_model"], rel_tol=0, abs_tol=1e-12)
        sites = ["a", "m", "h", "z"] if family.startswith("A4") else ["a", "m", "h", "q_post", "k_post", "v", "z"]
        assert set(condition["active_sites"]) == set(sites)
        pressure = ["h"] if family.endswith("-H") else (sites if "OL1" in family else [])
        if family == "A4-OL1-H":
            audited = next(r for r in audit["series"] if r["series_id"] == "run012_h_only" and r["attempt_id"] == attempt.name)
            assert audited["realized_pressure_sites"] == pressure
            assert audited["final_validation_loss"] == ordinary["loss"]
        else:
            assert set(condition["pressure_sites"]) == set(pressure)
        if pressure:
            assert condition["pressure_method"] == "orthogonal_l1"
            assert condition["pressure_weight"] == condition["step_budget"] == 1
            if family != "A4-OL1-H":
                assert verified["ol1"]["pressure_capture_tensor_count"] == 6 * len(pressure)
        else:
            assert condition["pressure_method"] == "none"
        loss_pass = "ordinary_final" if family.endswith("-H") else "logical_diagnostic_eager"
        assert original["loss"] == (ordinary["loss"] if loss_pass == "ordinary_final" else eager["loss"])
        rows.append({
            "family": family, "label": FAMILIES[family], "kappa": original["dose"],
            "S_model_percent": 100 * ratio, "R_model_fraction": ratio, **counts,
            "reported_validation_loss": original["loss"], "reported_loss_pass": loss_pass,
            "ordinary_final_validation_loss": ordinary["loss"],
            "logical_diagnostic_validation_loss": eager["loss"],
            "logical_minus_ordinary_loss": eager["loss"] - ordinary["loss"],
            "run": run.relative_to(ROOT).as_posix(), "attempt_id": attempt.name,
            "checkpoint_content_sha256": verified["checkpoint_content_sha256"],
            "gate_sites": sites, "realized_pressure_sites": pressure,
            "declared_pressure_sites": condition["pressure_sites"],
            "source_attempt": attempt.relative_to(ROOT).as_posix(),
        })
    assert len(rows) == 30 and len(identities) == 1
    for family in FAMILIES:
        assert sorted(r["kappa"] for r in rows if r["family"] == family) == KAPPAS
    rows.sort(key=lambda r: (r["kappa"], list(FAMILIES).index(r["family"])))
    return {
        "status": "verified_30_endpoints", "user_approved_for_paper_use": "2026-09-17",
        "sources_sha256": sources, "coverage": frozen["coverage"],
        "matched_initialization_and_schedule_sha256": list(next(iter(identities))),
        "training_steps_per_condition": 712, "training_input_tokens_per_condition": 1493172224,
        "seed": 1234, "a4_h_only_realization_audit": realization,
        "metric": "S_model_percent = 100 * R_model; count-pooled logical opportunity, not runtime speedup",
        "loss_provenance_correction": "The previously delivered table mixes final-checkpoint evaluation passes. Preserve its values exactly; retain both uniform-pass alternatives explicitly.",
        "maximum_absolute_logical_minus_ordinary_loss": max(abs(r["logical_minus_ordinary_loss"]) for r in rows),
        "rows": rows,
    }


def write_table(data, field, title, note, filename):
    lines = [f"# {title}", "", note, "", "Each cell is (S_model %, validation loss in nats).", "",
             "| kappa | " + " | ".join(FAMILIES.values()) + " |",
             "|---:|" + "---:|" * len(FAMILIES)]
    for kappa in KAPPAS:
        rows = [r for r in data["rows"] if r["kappa"] == kappa]
        cells = [f"({r['S_model_percent']:.4f}, {r[field]:.6f})" for r in rows]
        lines.append(f"| {kappa:g} | " + " | ".join(cells) + " |")
    lines.extend(["", "All 30 points: seed 1234, 712 updates, 1,493,172,224 training tokens;",
                  "500 validation documents, 338 complete blocks, 1,444-token excluded tail.",
                  "A4+OL1(h) is the audited realized Run 012 intervention. See README.md and results.json.", ""])
    (HERE / filename).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    data = build()
    (HERE / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_table(data, "reported_validation_loss", "Approved 14M table — exact delivered values",
                "Archival table: A4/A7 without pressure and with all-site OL1 use eager logical-pass loss; h-only columns use ordinary final validation loss. All evaluate the final checkpoint.", "TABLE.md")
    write_table(data, "ordinary_final_validation_loss", "14M table — uniform ordinary final validation",
                "Explicit alternative: all six columns use ordinary final-checkpoint validation loss.", "TABLE-ordinary-final.md")
    write_table(data, "logical_diagnostic_validation_loss", "14M table — uniform eager logical diagnostic",
                "Explicit alternative: all six columns pair loss and sparsity from the same final-checkpoint logical diagnostic pass, matching the older paper overview's convention.", "TABLE-logical-pass.md")
    print(f"Verified {len(data['rows'])} endpoints; largest loss-pass difference = {data['maximum_absolute_logical_minus_ordinary_loss']:.12f}")


if __name__ == "__main__":
    main()
