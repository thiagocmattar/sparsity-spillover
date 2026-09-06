"""K012 frozen-candidate Pythia-14M development probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent


def create_adapter(model, implementation, sites):
    if implementation != "k012":
        raise ValueError("This development probe accepts only K012")
    source = HERE / "candidates/k012/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k012", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return base.SiteAdapter(module.Adapter(model), implementation, sites)


def source_records(implementation):
    if implementation != "k012":
        raise ValueError("This development probe accepts only K012")
    sources = [
        Path(__file__),
        HERE / "probe_models.py",
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k012/candidate.py",
        HERE / "candidates/k001/candidate.py",
        HERE / "candidates/k001/kernel.cu",
    ]
    sources += sorted((RUN.parents[1] / "src/sparsity_research").glob("*.py"))
    return [base.record(source) for source in sources]


base.IMPLEMENTATIONS = ("k012",)
base.create_adapter = create_adapter
base.source_records = source_records


if __name__ == "__main__":
    base.main()
