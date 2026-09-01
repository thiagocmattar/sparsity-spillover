"""Run 018 non-evidence probes using the canonical initialization artifact."""

from initialization_artifact import load_pinned_initialization
from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_smoke", "smoke.py")
_FROZEN.apply_pythia_70m_initialization = load_pinned_initialization
run_smoke = _FROZEN.run_smoke
