"""Run 032 worker execution with audited h-only pressure capture."""

from _reuse_run004 import load_run004_module
from run032_capture import HOnlyPressureCapture


_BASE = load_run004_module("_run032_frozen_run004_training", "training.py")
_BASE.ActivationCapture = HOnlyPressureCapture

run_worker = _BASE.run_worker
run_condition = _BASE.run_condition
timed_validation = _BASE.timed_validation
