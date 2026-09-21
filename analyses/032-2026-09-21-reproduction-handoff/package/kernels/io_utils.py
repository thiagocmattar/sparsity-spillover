from support import ROOT, module, sha256, read_json

__all__ = ["RUN", "module", "read", "verify"]
RUN = ROOT / "70m"
read = read_json


def verify(row):
    path = RUN / row["path"]
    if not path.resolve().is_relative_to(RUN.resolve()):
        raise ValueError("Unsafe kernel source path")
    if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
        raise ValueError(f"Frozen kernel changed: {path}")
    return path
