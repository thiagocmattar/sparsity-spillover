"""K004 flagged development probe fixed to the z projection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STRATEGY = "flagged"
FIXED_SITES = frozenset(("z",))


def select_sites(selection, topology):
    if selection != "active":
        raise ValueError("The flagged-z probe requires --sites active")
    return FIXED_SITES


def create_adapter(model, implementation, sites):
    if implementation != "k004" or frozenset(sites) != FIXED_SITES:
        raise ValueError("This probe accepts only flagged K004 at z")
    source = HERE / "candidates/k004/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k004_flagged_z", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return base.SiteAdapter(
        module.Adapter(model, strategy=STRATEGY), implementation, sites
    )


def source_records(implementation):
    if implementation != "k004":
        raise ValueError("This probe accepts only K004")
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
