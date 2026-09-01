"""Freeze Run 017's uniform TEAL protocol against Run 018 checkpoints."""

from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_teal_posthoc", "teal_posthoc.py")
for _name in dir(_FROZEN):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_FROZEN, _name)
