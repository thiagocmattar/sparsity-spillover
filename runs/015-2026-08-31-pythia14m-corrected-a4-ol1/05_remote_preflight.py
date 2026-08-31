#!/usr/bin/env python
"""Execute the matched A4 preflight through Run 015's corrected modules."""

from __future__ import annotations

import runpy

from _reuse_run012 import RUN012_DIR


if __name__ == "__main__":
    runpy.run_path(str(RUN012_DIR / "05_remote_preflight.py"), run_name="__main__")
