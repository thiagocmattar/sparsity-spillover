"""Run-local immutable source bindings and file identity helpers."""
from pathlib import Path
import importlib.util
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
R25 = ROOT/'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search'
R26 = ROOT/'runs/026-2026-09-06-pythia14m-fused-sparse-kernel'
sys.path[:0] = [str(R25),str(ROOT/'src')]
from run025_common import read_json, write_json, sha256, record, verify_record


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dense = module('run027_dense_probe', R25/'autoresearch/dense_probe/probe.py')


def inputs():
    manifest = read_json(RUN/'prelaunch/inputs.json')
    for row in manifest['inputs'].values(): verify_record(row)
    return manifest
