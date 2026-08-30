"""Run 012 worker execution using frozen Run 004 full-pass orchestration."""

from _reuse_run004 import load_run004_module


_BASE = load_run004_module("_run012_frozen_run004_training", "training.py")

run_worker = _BASE.run_worker
run_condition = _BASE.run_condition
timed_validation = _BASE.timed_validation
