"""The paper's executed 95-condition grid, separated from deployment choices."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGIES = {"0": "A0", "1": "A1-H", "hz": "HZ", "4": "A4-Z", "7": "A7-Z-POST"}
SITES = {
    "0": [],
    "1": ["h"],
    "hz": ["h", "z"],
    "4": ["a", "m", "h", "z"],
    "7": ["a", "m", "h", "q_post", "k_post", "v", "z"],
}
REVISIONS = {
    "14M": "7386d9a4ae45aef494a6e704910394def3037fc5",
    "31M": "b7782556ba7adfb4730d9bda7d12aa44d88fa132",
    "70M": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
    "410M": "b5e8535141902c0e985cea61fd02afe7fe86af32",
}


def register_hz():
    from sparsity_research.sites import TOPOLOGIES, Topology

    TOPOLOGIES["HZ"] = Topology("HZ", ("h", "z"))


def paper_rows():
    return json.loads((ROOT / "results/endpoints.json").read_text())["trained_points"]


def identifier(row):
    p = {"none": "P0", "h": "Ph", "all": "Pall", "L1": "L1"}[row["pressure"]]
    dose = (
        row["local_pressure_weight"]
        if row["scope"] == "1" and row["pressure"] != "none"
        else row["kappa"]
    )
    return f"{row['model']}-T{'2' if row['scope'] == 'hz' else row['scope']}-{p}" + (
        f"-{dose:g}" if dose is not None else ""
    )


def resolve(identifier_):
    row = next((r for r in paper_rows() if identifier(r) == identifier_), None)
    if row is None:
        raise ValueError("Unknown paper condition; use python -m training.train --list")
    size = row["model"]
    scope = row["scope"]
    pressure = row["pressure"]
    kappa = row["kappa"] or 0.0
    gates = {
        s: {
            "operator": "symmetric_threshold"
            if s in ("q_post", "k_post", "v")
            else "one_sided_threshold",
            "kappa": kappa,
        }
        for s in SITES[scope]
    }
    if scope == "1":
        gates = {"h": {"operator": "relu"}}
    model = {
        "architecture": f"EleutherAI/pythia-{size.lower()}-deduped",
        "revision": REVISIONS[size],
        "initialization": "random",
        "topology_id": TOPOLOGIES[scope],
        "site_gates": gates,
    }
    sites = (
        [] if pressure == "none" else ["h"] if pressure in ("h", "L1") else SITES[scope]
    )
    method = (
        "none" if not sites else "l1_naive" if pressure == "L1" else "orthogonal_l1"
    )
    weight = 0.0 if not sites else row["local_pressure_weight"] if scope == "1" else 1.0
    peak = 3e-4 if size == "410M" else 1e-3
    reference = next(
        r
        for r in paper_rows()
        if r["model"] == size
        and r.get("initial_parameter_sha256")
        and r.get("training_schedule_hash")
    )
    return dict(
        id=identifier_,
        model=model,
        size=size,
        seed=1234,
        pressure=dict(
            method=method,
            sites=sites,
            weight=weight,
            step_budget=1.0 if method == "orthogonal_l1" else None,
        ),
        training=dict(
            max_steps=712,
            micro_batch_size=32 if size == "14M" else 4,
            gradient_accumulation_steps=32 if size == "14M" else 256,
            peak_learning_rate=peak,
            minimum_learning_rate=peak / 10,
            warmup_fraction=0.01,
            optimizer="adamw",
            adamw_betas=[0.9, 0.95],
            adamw_eps=1e-8,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
        ),
        expected_initial_parameter_sha256=row.get(
            "initial_parameter_sha256", reference["initial_parameter_sha256"]
        ),
        expected_schedule_sha256=row.get(
            "training_schedule_hash", reference["training_schedule_hash"]
        ),
        condition=identifier_,
    )


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        tensor = p.detach().contiguous().cpu()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(json.dumps(list(tensor.shape)).encode("ascii"))
        h.update(tensor.numpy().tobytes(order="C"))
    return h.hexdigest()
