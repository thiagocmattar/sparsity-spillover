"""Frozen K013 evaluator built on the immutable all-size final evaluator."""

from pathlib import Path

import probe_frozen_final as base


base.SPECS = {
    "k013": {
        "size": "14m",
        "freeze": "candidates/k013/FROZEN.json",
        "candidate": "candidates/k013/candidate.py",
        "parents": ("candidates/k001/candidate.py", "candidates/k001/kernel.cu"),
        "site_policy": "topology",
    }
}
base.IMPLEMENTATIONS = ("k013",)
_base_source_records = base.source_records


def source_records(implementation):
    return [base.record(Path(__file__))] + _base_source_records(implementation)


base.source_records = source_records


if __name__ == "__main__":
    base.main()
