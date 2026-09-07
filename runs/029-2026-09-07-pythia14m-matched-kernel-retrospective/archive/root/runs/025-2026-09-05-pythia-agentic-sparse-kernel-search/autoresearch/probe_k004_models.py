"""K004-inline development model probe using the frozen v1 probe machinery.

The first model-probe source has already produced K001 artifacts and remains
unchanged.  This append-only entry point fixes K004's development strategy to
``inline`` and records itself plus both K004 source files in every artifact.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STRATEGY = "inline"


def create_adapter(model, implementation, sites):
    if implementation != "k004":
        raise ValueError("This append-only probe accepts only K004")
    source = HERE / "candidates/k004/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k004_inline", source)
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
base.create_adapter = create_adapter
base.source_records = source_records


if __name__ == "__main__":
    base.main()
