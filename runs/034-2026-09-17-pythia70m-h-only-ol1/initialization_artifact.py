"""Reuse the verified Run018 canonical loader, with Run034 paths/config."""
from _reuse_run018 import load_run018_module
_BASE = load_run018_module("_run034_frozen_run018_artifact", "initialization_artifact.py")
load_pinned_initialization = _BASE.load_pinned_initialization
