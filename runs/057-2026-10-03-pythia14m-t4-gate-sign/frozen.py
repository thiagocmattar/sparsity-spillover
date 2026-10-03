"""Load frozen Run 004 modules under private names for audited code reuse."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


RUN004_DIR = Path(__file__).resolve().parent.parent / "004-2026-08-29-pythia14m-full-pass-l1n"


def load_run004_module(alias: str, filename: str) -> ModuleType:
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    path = RUN004_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen Run 004 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module
