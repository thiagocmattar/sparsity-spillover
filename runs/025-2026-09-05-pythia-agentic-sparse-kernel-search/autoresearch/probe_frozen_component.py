"""Shared policy wiring for frozen FFN/projection contribution probes.

The executed evaluator remains ``probe_frozen_final.py``: this module only
restricts its already-frozen adapter to a predeclared subset of eligible linear
sites. QK/PV attention and the dense LM head remain unchanged.
"""

from __future__ import annotations

from pathlib import Path


SPECS = {
    "k013": {
        "size": "14m",
        "freeze": "candidates/k013/FROZEN.json",
        "candidate": "candidates/k013/candidate.py",
        "parents": ("candidates/k001/candidate.py", "candidates/k001/kernel.cu"),
        "site_policy": "topology",
    },
    "k016": {
        "size": "70m",
        "freeze": "candidates/k016/FROZEN.json",
        "candidate": "candidates/k016/candidate.py",
        "parents": ("candidates/k001/candidate.py", "candidates/k001/kernel.cu"),
        "site_policy": "topology",
    },
    "k010": {
        "size": "410m",
        "freeze": "candidates/k010/FROZEN.json",
        "candidate": "candidates/k010/candidate.py",
        "parents": ("candidates/k004/candidate.py", "candidates/k004/kernels.py"),
        "site_policy": "z_only",
    },
}

GROUPS = {
    "ffn": frozenset(("m", "h")),
    "attention_projection": frozenset(("a", "z")),
}

K016_ALLOWED = {
    "A0": frozenset(),
    "A1-H": frozenset(("h",)),
    "A4-Z": frozenset(("a", "m", "h", "z")),
    "A7-Z-POST": frozenset(("h", "z")),
}


def frozen_allowed_sites(implementation: str, topology: dict) -> frozenset[str]:
    topology_id = topology["topology_id"]
    if implementation == "k013":
        if topology_id == "A0":
            return frozenset(("a", "m", "h", "z"))
        return frozenset(("a", "m", "h", "z")).intersection(topology["active_sites"])
    if implementation == "k016":
        if topology_id not in K016_ALLOWED:
            raise ValueError(f"Unsupported K016 topology: {topology_id}")
        return K016_ALLOWED[topology_id]
    if implementation == "k010":
        return frozenset(("z",))
    raise ValueError(f"Unsupported frozen implementation: {implementation}")


def component_sites(component: str, implementation: str, topology: dict) -> frozenset[str]:
    if component not in GROUPS:
        raise ValueError(f"Unsupported component: {component}")
    selected = GROUPS[component].intersection(frozen_allowed_sites(implementation, topology))
    if not selected:
        raise ValueError(
            f"Frozen {implementation}/{topology['topology_id']} has no {component} sparse sites"
        )
    return selected


def run(component: str, wrapper_path: str) -> None:
    import probe_frozen_final as base

    base.SPECS = SPECS
    base.IMPLEMENTATIONS = tuple(SPECS)
    original_source_records = base.source_records

    def select_sites(selection, topology, implementation):
        if selection != "active":
            raise ValueError("Component probes require --sites active")
        return component_sites(component, implementation, topology)

    def source_records(implementation):
        return [
            base.record(Path(wrapper_path)),
            base.record(Path(__file__)),
        ] + original_source_records(implementation)

    base.select_sites = select_sites
    base.source_records = source_records
    base.main()
