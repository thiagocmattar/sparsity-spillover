#!/usr/bin/env python3
"""Emit one Run-024 progress/ETC/GPU/cost snapshot."""

from _reuse_run023 import load_run023_module


_impl = load_run023_module("_run023_monitor", "06_monitor.py")


if __name__ == "__main__":
    _impl.main()

