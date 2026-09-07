"""K008 bounded layer-mask development probe using frozen v1 machinery."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import probe_models as base


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
MASK_ENV = "RUN025_K008_MASK"


def candidate_module():
    source = HERE / "candidates/k008/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_probe_k008", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_mask(module):
    identifier = os.environ.get(MASK_ENV)
    module.selected_layers(identifier)
    return identifier


def create_adapter(model, implementation, sites):
    if implementation != "k008":
        raise ValueError("This append-only probe accepts only K008")
    module = candidate_module()
    return base.SiteAdapter(
        module.Adapter(model, mask_id=selected_mask(module)), implementation, sites
    )


def source_records(implementation):
    if implementation != "k008":
        raise ValueError("This append-only probe accepts only K008")
    sources = [
        Path(__file__),
        HERE / "probe_models.py",
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k008/candidate.py",
        HERE / "candidates/k001/candidate.py",
        HERE / "candidates/k001/kernel.cu",
    ]
    sources += sorted((RUN.parents[1] / "src/sparsity_research").glob("*.py"))
    return [base.record(source) for source in sources]


base.IMPLEMENTATIONS = ("k008",)
base.create_adapter = create_adapter
base.source_records = source_records


if __name__ == "__main__":
    base.main()
