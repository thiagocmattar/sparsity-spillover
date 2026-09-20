"""Derive Table 1's h,z reach ceilings from retained architecture counts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    data = json.loads((HERE / "data/full-trained-results.json").read_text())
    rows = []
    for size in ("14M", "70M", "410M"):
        endpoint = next(r for r in data["trained_points"] if r["model"] == size)
        source = ROOT / endpoint["source_attempt"] / "diagnostics/logical_products.json"
        ceiling = json.loads(source.read_text())["architecture_maximum"]
        ceiling["topology_id"] = "T2-hz"
        ceiling["active_sites"] = ["h", "z"]
        ceiling["reachable_operations"] = ["mlp_w2", "attention_output_projection"]
        numerator = ceiling["layers"] * sum(
            ceiling["per_block_operation_products"][op]
            for op in ceiling["reachable_operations"]
        )
        L, T, d, f, V = (ceiling[k] for k in (
            "layers", "sequence_length", "hidden_size", "ffn_size", "vocabulary_size"
        ))
        assert numerator == L * T * (f * d + d * d)
        denominator = ceiling["model_product_count"]
        assert denominator == L * (T * (4*d*d + 2*d*f) + d*T*(T+1)) + T*d*V
        ceiling["reachable_product_count"] = numerator
        ceiling["R_model_max_fraction"] = numerator / denominator
        ceiling["R_model_max_percent"] = 100 * numerator / denominator
        rows.append(dict(model=size, source=source.relative_to(ROOT).as_posix(),
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         unit="scalar products per full uncached sequence", **ceiling))
    expected = json.loads((HERE / "data/figure-data.json").read_text())["ceilings"]["2"]
    assert rows[0]["R_model_max_fraction"] == expected["R_model_max_fraction"]
    result = dict(recipe="T2/Ph", pressure_sites=["h"],
                  interpretation="Analytic reach ceilings, not measured sparsity or speedup.",
                  rows=rows)
    (HERE / "data/table1-t2-ceilings.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(" & ".join(f"{r['R_model_max_percent']:.2f}" for r in rows))


if __name__ == "__main__":
    main()
