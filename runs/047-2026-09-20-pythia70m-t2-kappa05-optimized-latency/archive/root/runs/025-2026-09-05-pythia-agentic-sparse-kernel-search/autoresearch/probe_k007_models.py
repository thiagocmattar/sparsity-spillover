"""K007 bounded-geometry development probe using the frozen v1 machinery."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
CONFIG_ENV = "RUN025_K007_CONFIG"


def candidate_module():
    source = HERE / "candidates/k007/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k007", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_config(module):
    identifier = os.environ.get(CONFIG_ENV)
    module.geometry(identifier)
    return identifier


def create_adapter(model, implementation, sites):
    if implementation != "k007":
        raise ValueError("This append-only probe accepts only K007")
    module = candidate_module()
    return base.SiteAdapter(
        module.Adapter(model, config_id=selected_config(module)),
        implementation,
        sites,
    )


def source_records(implementation):
    if implementation != "k007":
        raise ValueError("This append-only probe accepts only K007")
    sources = [
        Path(__file__),
        HERE / "probe_models.py",
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k007/candidate.py",
        HERE / "candidates/k007/kernels.py",
    ]
    sources += sorted((RUN.parents[1] / "src/sparsity_research").glob("*.py"))
    return [base.record(source) for source in sources]


base.IMPLEMENTATIONS = ("k007",)
base.create_adapter = create_adapter
base.source_records = source_records


if __name__ == "__main__":
    base.main()
