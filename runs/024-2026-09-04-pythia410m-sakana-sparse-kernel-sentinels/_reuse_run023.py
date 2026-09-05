"""Load frozen Run-023 modules under private names."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


RUN023_DIR = Path(__file__).resolve().parent.parent / "023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels"


def load_run023_module(alias: str, filename: str) -> ModuleType:
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    path = RUN023_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen Run 023 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module

