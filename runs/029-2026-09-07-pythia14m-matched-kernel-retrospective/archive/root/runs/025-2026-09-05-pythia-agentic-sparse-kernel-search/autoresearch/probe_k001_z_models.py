"""K001 development probe fixed to the attention-output z projection."""

from __future__ import annotations

from pathlib import Path

import probe_models as base


FIXED_SITES = frozenset(("z",))
BASE_SOURCE_RECORDS = base.source_records


def select_sites(selection, topology):
    if selection != "active":
        raise ValueError("The z-only probe requires --sites active")
    return FIXED_SITES


def source_records(implementation):
    if implementation != "k001":
        raise ValueError("This append-only probe accepts only K001")
    return [base.record(Path(__file__)), *BASE_SOURCE_RECORDS(implementation)]


base.IMPLEMENTATIONS = ("k001",)
base.select_sites = select_sites
base.source_records = source_records


if __name__ == "__main__":
    base.main()
