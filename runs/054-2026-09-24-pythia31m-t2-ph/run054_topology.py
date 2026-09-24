"""Paper T2 is operational HZ, independent of historical A2."""
from sparsity_research.sites import TOPOLOGIES, Topology


def register():
    existing = TOPOLOGIES.get("HZ")
    if existing is not None and existing.active_sites != ("h", "z"):
        raise RuntimeError("Conflicting HZ registration")
    TOPOLOGIES["HZ"] = Topology("HZ", ("h", "z"))
