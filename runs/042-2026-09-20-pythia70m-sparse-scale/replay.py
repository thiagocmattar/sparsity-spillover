"""Native and frozen scale comparators plus immutable optimization candidates."""
from frozen_replay import R25, R26, R27, R28, dense, scaffold
from io_utils import RUN, module, read, verify
from controls import MODES, original_norms, substitute


def install_candidate(model, identifier):
    manifest = read(RUN / 'candidates' / identifier / 'manifest.json')
    for item in manifest['files']:
        path = verify(item)
        assert path.is_relative_to(RUN / 'candidates' / identifier)
    implementation = module('run042_' + identifier, RUN/'candidates'/identifier/'candidate.py')
    return implementation.install(model)


def install(model, mode):
    if mode not in MODES:
        return install_candidate(model, mode)
    if mode == 'native':
        return {'mode': mode, 'implementation': 'unchanged native PyTorch'}
    saved = original_norms(model)
    if mode.startswith('selected'):
        selection = read(RUN/'provenance/final-selection.json')
        verify(selection['manifest'])
        metadata = install_candidate(model, selection['candidate'])
        if mode == 'selected-no-skip':
            for layer in model.gpt_neox.layers:
                layer._run026_joint.skip = False
        elif mode != 'selected':
            substitute(model, mode.removeprefix('selected-'), saved)
        return {**metadata, 'ablation': mode}
    if model.config.hidden_size == 128:
        original = module('run042_14m_adapter', R27/'adapter.py')
        original.install(model, 'sparse')
        frozen = module('run042_14m_k050', R28/'candidates/k050/candidate.py')
        metadata = frozen.install(model, shortcut=False, round_p=False,
                                  skip=mode != 'all-skips-off', projection_skip=mode != 'all-skips-off')
    elif mode == 'legacy':
        frozen = module('run042_legacy70', RUN/'kernel/candidate.py')
        return frozen.install(model, shortcut=False, round_p=False, skip=True, projection_skip=True)
    else:
        baseline = module('run042_baseline70', RUN/'base70/candidate.py')
        metadata = baseline.install(model)
        if mode == 'all-skips-off':
            for layer in model.gpt_neox.layers:
                layer._run026_joint.skip = False
    if mode not in ('full', 'all-skips-off'):
        substitute(model, mode, saved)
    return {**metadata, 'mode': mode, 'weights_and_gates_unchanged': True}
