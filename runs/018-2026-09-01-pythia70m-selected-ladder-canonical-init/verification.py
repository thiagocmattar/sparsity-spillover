"""Freeze Run 017's attempt and cohort verification against Run 018 identities."""

from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_verification", "verification.py")
for _name in dir(_FROZEN):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_FROZEN, _name)
