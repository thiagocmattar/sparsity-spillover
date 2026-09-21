"""The approved h/z topology is local to Run041; historical registries stay frozen."""
def register():
    from sparsity_research.sites import TOPOLOGIES, Topology
    desired=Topology("HZ", ("h", "z"))
    if "HZ" in TOPOLOGIES and TOPOLOGIES["HZ"] != desired:
        raise RuntimeError("Conflicting HZ topology")
    TOPOLOGIES["HZ"]=desired
