"""Run 015 worker execution with audited four-site pressure capture."""

from _reuse_run004 import load_run004_module
from run015_capture import FourSitePressureCapture


_BASE = load_run004_module("_run015_frozen_run004_training", "training.py")
_BASE.ActivationCapture = FourSitePressureCapture

run_worker = _BASE.run_worker
run_condition = _BASE.run_condition
timed_validation = _BASE.timed_validation
