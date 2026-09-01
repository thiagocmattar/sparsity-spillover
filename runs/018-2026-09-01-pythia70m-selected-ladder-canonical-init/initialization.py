"""Frozen Pythia-70M recipe initialization used only to generate the canonical artifact."""

from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_initialization", "initialization.py")
EXPECTED_ARCHITECTURE = _FROZEN.EXPECTED_ARCHITECTURE
RESIDUAL_OUTPUT_SUFFIXES = _FROZEN.RESIDUAL_OUTPUT_SUFFIXES
apply_pythia_70m_initialization = _FROZEN.apply_pythia_70m_initialization
apply_pythia_14m_initialization = apply_pythia_70m_initialization
verify_recipe_model = _FROZEN.verify_recipe_model
