"""Frozen Run 004 FP16 AdamW boundary; Run 013 has no activation pressure."""

from _reuse_run004 import load_run004_module


_BASE = load_run004_module("_run013_frozen_run004_optimizer_boundary", "optimizer_boundary.py")

DynamicLossScaler = _BASE.DynamicLossScaler
build_recipe_adamw = _BASE.build_recipe_adamw
recipe_attention_context = _BASE.recipe_attention_context
recipe_learning_rate = _BASE.recipe_learning_rate
run_recipe_boundary = _BASE.run_recipe_boundary
