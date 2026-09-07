"""K004 flagged-tile development probe using the frozen v1 machinery.

This append-only entry point reuses one activity flag per 16x32 input tile
across every output-N program.  It exists to test whether that extra launch
and flag traffic amortize at the wider 70M/410M projection shapes.  The
candidate math, checkpoints, inputs, gates, and timing protocol are unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STRATEGY = "flagged"


def create_adapter(model, implementation, sites):
    if implementation != "k004":
        raise ValueError("This append-only probe accepts only K004")
    source = HERE / "candidates/k004/candidate.py"
    spec = importlib.util.spec_from_file_location(
        "run025_probe_k004_flagged", source
    )
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
