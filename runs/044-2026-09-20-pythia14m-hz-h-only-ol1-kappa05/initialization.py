"""Replay the approved random step-zero bytes across GPU RNG implementations."""

import hashlib
import json
from pathlib import Path

from _reuse_run004 import load_run004_module


_BASE = load_run004_module("_run044_frozen_run004_initialization", "initialization.py")

verify_recipe_model = _BASE.verify_recipe_model


def apply_pythia_14m_initialization(model, *, torch):
    from safetensors.torch import load_file
    from run_config import EXPECTED_INITIAL_PARAMETER_SHA256, parameter_sha256
    metadata = _BASE.apply_pythia_14m_initialization(model, torch=torch)
    root = Path(__file__).resolve().parent / 'inputs/random-initialization'
    provenance = json.loads((root / 'provenance.json').read_text())
    path = root / 'model.safetensors'
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != provenance['file_sha256'] or provenance['trained_updates'] != 0:
        raise ValueError('Random initialization snapshot identity mismatch')
    model.load_state_dict(load_file(path, device='cpu'), strict=True)
    if parameter_sha256(model) != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise ValueError('Approved initial parameters were not restored exactly')
    metadata['random_initialization_replay'] = provenance
    model.config.pythia_recipe_initialization['random_initialization_replay'] = provenance
    return metadata
