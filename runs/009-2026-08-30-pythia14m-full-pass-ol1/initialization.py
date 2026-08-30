"""Frozen Run 004 Pythia recipe initialization, reused without modification."""

from _reuse_run004 import load_run004_module


_BASE = load_run004_module("_run009_frozen_run004_initialization", "initialization.py")

apply_pythia_14m_initialization = _BASE.apply_pythia_14m_initialization
verify_recipe_model = _BASE.verify_recipe_model
