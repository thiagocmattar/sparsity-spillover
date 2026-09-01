"""Freeze Run 017's verified FP16 control and condition-driven OL1 boundary."""

from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_optimizer_boundary", "optimizer_boundary.py")
for _name in dir(_FROZEN):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_FROZEN, _name)
