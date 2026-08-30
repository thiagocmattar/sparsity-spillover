#!/usr/bin/env python
"""Run the frozen, hash-locked Run 004 MiniPile cache builder."""

from __future__ import annotations

import runpy

from _reuse_run004 import RUN004_DIR


if __name__ == "__main__":
    runpy.run_path(str(RUN004_DIR / "06_build_cache_from_hf.py"), run_name="__main__")
