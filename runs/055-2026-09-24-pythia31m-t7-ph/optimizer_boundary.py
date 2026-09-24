"""Reuse the executed Base and audited h-only OL1 boundaries unchanged."""
import importlib.util
from pathlib import Path
import sys
from _reuse_run004 import load_run004_module

BASE = load_run004_module("_run055_base_boundary", "optimizer_boundary.py")
path = Path(__file__).resolve().parent.parent / "043-2026-09-20-pythia70m-hz-h-only-ol1/optimizer_boundary.py"
spec = importlib.util.spec_from_file_location("_run055_ol1_boundary", path)
OL1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = OL1
spec.loader.exec_module(OL1)
DynamicLossScaler = BASE.DynamicLossScaler
build_recipe_adamw = BASE.build_recipe_adamw
recipe_attention_context = BASE.recipe_attention_context
recipe_learning_rate = BASE.recipe_learning_rate


def run_recipe_boundary(**kwargs):
    pressure = kwargs["pressure"]
    if pressure.method == "none":
        return BASE.run_recipe_boundary(**kwargs)
    if not pressure.orthogonal:
        raise ValueError("Run055 permits Base or h-only OL1")
    return OL1.run_recipe_boundary(**kwargs)
