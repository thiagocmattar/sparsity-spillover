"""K014 exploratory 70M mask probe using complete-validation v1 machinery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PLAN = HERE / "candidates/k014/SEARCH.json"
IMPLEMENTATIONS = tuple(f"k014-{mask}" for mask in json.loads(PLAN.read_text())["masks"])


def parse_implementation(implementation):
    prefix = "k014-"
    if implementation not in IMPLEMENTATIONS or not implementation.startswith(prefix):
        raise ValueError("Implementation is outside the predeclared K014 search")
    return implementation[len(prefix) :]


def development_condition(manifest, identifier):
    row = next((item for item in manifest["checkpoints"] if item["id"] == identifier), None)
    allowed = set(json.loads(PLAN.read_text())["partition_boundary"]["development_conditions"])
    if (
        row is None
        or row.get("size") != "70m"
        or row.get("partition") != "development"
        or identifier not in allowed
    ):
        raise ValueError("K014 accepts only predeclared 70M development conditions")
    return row


def candidate_module():
    source = HERE / "candidates/k014/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k014", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_adapter(model, implementation, sites):
    mask = parse_implementation(implementation)
    module = candidate_module()
    adapter = module.Adapter(model, mask_id=mask)
    if adapter.implementation != implementation:
        raise ValueError("K014 implementation/mask identity mismatch")
    return base.SiteAdapter(adapter, implementation, sites)


def source_records(implementation):
    parse_implementation(implementation)
    sources = [
        Path(__file__),
        PLAN,
        HERE / "probe_models.py",
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k014/candidate.py",
        HERE / "candidates/k001/candidate.py",
        HERE / "candidates/k001/kernel.cu",
    ]
    sources += sorted((RUN.parents[1] / "src/sparsity_research").glob("*.py"))
    return [base.record(source) for source in sources]


base.IMPLEMENTATIONS = IMPLEMENTATIONS
base.development_condition = development_condition
base.create_adapter = create_adapter
base.source_records = source_records


if __name__ == "__main__":
    base.main()
