"""Frozen 70M port plus individually identified execution controls."""
from frozen_replay import R25, R26, R27, R28, dense, scaffold
from io_utils import RUN, module, read, verify
from controls import MODES, original_norms, substitute


def install(model, mode):
    if mode not in MODES:
        if not (mode.startswith('opt') and mode[3:].isdigit() and len(mode) == 6):
            raise ValueError(mode)
        manifest = read(RUN / 'candidates' / mode / 'manifest.json')
        for record in manifest['files']:
            path = verify(record)
            if not path.is_relative_to(RUN / 'candidates' / mode):
                raise ValueError('Candidate source outside its immutable directory')
        implementation = module('run040_' + mode, RUN / 'candidates' / mode / 'candidate.py')
        return implementation.install(model)
    if mode == "native":
        return {"mode": mode, "implementation": "unchanged native PyTorch"}
    saved = original_norms(model)
    frozen = module("run040_frozen_candidate", RUN / "kernel/candidate.py")
    enabled = mode != "all-skips-off"
    metadata = frozen.install(model, shortcut=False, round_p=False,
                              skip=enabled, projection_skip=enabled)
    substitute(model, mode, saved)
    return {**metadata, "mode": mode, "weights_and_gates_unchanged": True}
