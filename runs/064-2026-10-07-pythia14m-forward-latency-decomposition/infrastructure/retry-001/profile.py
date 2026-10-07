"""Retry unchanged measurements with a shape adapter for untimed gate counts.

The original executable and kernel sources remain unchanged. Only the old
diagnostic helper's flattened matrices are given their required batch axis.
"""
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(os.environ['RUN064_BASE'])
sys.path[:0]=[str(BASE),str(ROOT)]
import kernel
original=kernel.gated_value
def gated(value,mode,threshold):
    if mode==2 and value.ndim==2:
        return original(value.unsqueeze(0),mode,threshold).squeeze(0)
    return original(value,mode,threshold)
kernel.gated_value=gated
spec=importlib.util.spec_from_file_location('run064_retry_profile',ROOT/'02_profile.py')
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
raise SystemExit(profile.main())
