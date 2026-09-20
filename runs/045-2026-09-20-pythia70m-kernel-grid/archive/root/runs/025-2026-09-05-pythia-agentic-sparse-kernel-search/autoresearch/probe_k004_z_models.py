"""K004 inline development probe fixed to the attention-output z projection.

The 70M occupancy pass showed that z has the strongest zero-tile structure in
both high-pressure A4 and A7 endpoints, while h-only K004 remained slower than
native.  This append-only probe tests a single, declared z-only dispatch policy
without changing the candidate math, inputs, gates, or timing protocol.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STRATEGY = "inline"
FIXED_SITES = frozenset(("z",))


def select_sites(selection, topology):
    if selection != "active":
        raise ValueError("The z-only probe requires --sites active")
    if "z" not in base.LINEAR_SITES:
        raise ValueError("z is not a declared linear site")
    return FIXED_SITES


def create_adapter(model, implementation, sites):
    if implementation != "k004" or frozenset(sites) != FIXED_SITES:
        raise ValueError("This append-only probe accepts only K004 at z")
    source = HERE / "candidates/k004/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k004_z", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return base.SiteAdapter(
        module.Adapter(model, strategy=STRATEGY), implementation, sites
    )


def source_records(implementation):
    if implementation != "k004":
        raise ValueError("This append-only probe accepts only K004")
    sources = [
        Path(__file__),
        HERE / "probe_models.py",
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k004/candidate.py",
        HERE / "candidates/k004/kernels.py",
    ]
    sources += sorted((RUN.parents[1] / "src/sparsity_research").glob("*.py"))
    return [base.record(source) for source in sources]


base.IMPLEMENTATIONS = ("k004",)
base.select_sites = select_sites
base.create_adapter = create_adapter
base.source_records = source_records


if __name__ == "__main__":
    base.main()
