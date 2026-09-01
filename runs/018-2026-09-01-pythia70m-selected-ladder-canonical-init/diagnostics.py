"""Freeze Run 017's scale-generic activation and logical-product diagnostics."""

from _reuse_run017 import load_run017_module


_FROZEN = load_run017_module("_run018_frozen_run017_diagnostics", "diagnostics.py")
AttentionOutputCapture = _FROZEN.AttentionOutputCapture
activation_diagnostic_validation = _FROZEN.activation_diagnostic_validation
logical_product_validation = _FROZEN.logical_product_validation
