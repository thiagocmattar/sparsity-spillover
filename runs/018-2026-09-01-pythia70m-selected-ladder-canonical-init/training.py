"""Run 018 lifecycle: frozen Run 017 science with canonical artifact realization."""

from __future__ import annotations

import sys

import run018_capture
from initialization_artifact import load_pinned_initialization
from _reuse_run017 import load_run017_module


# Frozen Run 017 imports this historical module name while loading. Bind it to
# the Run 018 implementation before executing the frozen source.
sys.modules.setdefault("run017_capture", run018_capture)
_FROZEN = load_run017_module("_run018_frozen_run017_training", "training.py")
_FROZEN._BASE.apply_pythia_14m_initialization = load_pinned_initialization

run_worker = _FROZEN.run_worker
run_condition = _FROZEN.run_condition
timed_validation = _FROZEN.timed_validation
_transfer_initialized_model_to_cuda = _FROZEN._transfer_initialized_model_to_cuda
_verified_initial_parameter_sha256 = _FROZEN._verified_initial_parameter_sha256
_build_random_pythia = _FROZEN._build_random_pythia
_microbatches_for_step = _FROZEN._microbatches_for_step
_save_checkpoint = _FROZEN._save_checkpoint
