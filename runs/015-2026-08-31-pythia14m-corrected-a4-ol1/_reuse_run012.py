"""Load Run 012 modules only where their declared A4 recipe remains valid."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


RUN012_DIR = Path(__file__).resolve().parent.parent / "012-2026-08-30-pythia14m-full-pass-a4-ol1"


def load_run012_module(alias: str, filename: str) -> ModuleType:
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    path = RUN012_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Run 012 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module
