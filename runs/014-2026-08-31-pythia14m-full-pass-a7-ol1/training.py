"""Run 014 worker execution with audited seven-site pressure capture."""

from _reuse_run004 import load_run004_module
from run014_capture import SevenSitePressureCapture


_BASE = load_run004_module("_run014_frozen_run004_training", "training.py")
_BASE.ActivationCapture = SevenSitePressureCapture

run_worker = _BASE.run_worker
run_condition = _BASE.run_condition
timed_validation = _BASE.timed_validation
