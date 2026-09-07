"""Run-local source bindings; preserve the sealed Run 025 helper files."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
SOURCE = ROOT / 'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search'
sys.path[:0] = [str(SOURCE), str(ROOT / 'src')]
from run025_common import read_json, write_json, sha256, record, verify_record
from measurement import numerical_gate, paired_schedule, timing_summary


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


dense = module('run026_inherited_dense', SOURCE / 'autoresearch/dense_probe/probe.py')


def inputs_manifest():
    manifest = read_json(SOURCE / 'prelaunch/input_manifest.json')
    if manifest['config_sha256'] != sha256(SOURCE / 'config.json'):
        raise ValueError('Source config identity changed')
    selected = read_json(RUN / 'config.json')['condition']
    row = next(row for row in manifest['checkpoints'] if row['id'] == selected)
    for item in row['files'] + row['provenance']:
        verify_record(item)
    return manifest, row
