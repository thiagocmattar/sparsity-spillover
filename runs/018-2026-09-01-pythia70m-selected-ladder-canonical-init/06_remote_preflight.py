#!/usr/bin/env python
"""Run 018 H200 gate: frozen Run 017 probe with canonical initialization loading."""

from initialization_artifact import load_pinned_initialization
from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_remote_preflight", "05_remote_preflight.py")
_FROZEN.apply_pythia_70m_initialization = load_pinned_initialization


if __name__ == "__main__":
    _FROZEN.main()
