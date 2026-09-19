"""CPU checks of the approved comparison and independent counter expectations."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controls = load('run037_test_controls', 'controls.py')


@pytest.mark.parametrize('op', controls.OPS)
def test_one_operation_changes(op):
    assert {key for key, active in controls.mask('without-' + op).items() if not active} == {op}
    assert all(controls.mask('full').values())
    assert controls.mask('frozen') == controls.mask('full')


def test_anchor_masks():
    assert not any(controls.mask('off').values())
    assert {key for key, value in controls.mask('projection').items() if value} == {'a', 'm', 'h', 'z'}
    assert len(set(controls.MODES)) == 10
    with pytest.raises(ValueError): controls.mask('without-q')


def test_signed_conditional_effects():
    values = dict.fromkeys(controls.MODES, 2.)
    values.update({'without-h': 3., 'without-qk': 1.5})
    effects = controls.contrasts(values)
    assert effects['h'] == {'saved_ms': 1., 'speedup': 1.5}
    assert effects['qk'] == {'saved_ms': -.5, 'speedup': .75}
    assert effects['m'] == {'saved_ms': 0., 'speedup': 1.}
    with pytest.raises(ValueError): controls.contrasts({'full': 2.})
    values['full'] = 0.
    with pytest.raises(ValueError): controls.contrasts(values)


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.setitem(sys.modules, 'common', SimpleNamespace(write_json=None))
    return load('run037_diagnostic_test', 'diagnostics.py')


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('pattern', ['zero', 'short', 'mixed', 'unsafe-small'])
def test_hybrid_counter_cases(diagnostic, skip, pattern):
    x = torch.zeros(16, 512)
    if pattern != 'zero': x[:, 0] = .75; x[:, 17] = -.5
    if pattern == 'mixed': x[8:, 32:48] = .625
    if pattern == 'unsafe-small': x[:, 0] = 2. ** -60
    expected = {'zero': [0, 1024, 0], 'short': [0, 1024, 4096],
                'mixed': [48, 976, 2048], 'unsafe-small': [64, 960, 0]}[pattern]
    assert diagnostic.hybrid_counts(x, True, skip) == (expected if skip else [1024, 0, 0])


def test_projection_fragments(diagnostic):
    x = torch.zeros(16, 128)
    x[:, 0] = 1.
    counts = diagnostic.projection_counts(x, 384)
    assert counts == {'product_count': 16 * 128 * 384, 'zero_product_count': (16 * 128 - 16) * 384,
                      'issued_mmas': 48, 'skipped_mmas': 336}


def test_approved_input_identity_and_coverage():
    cfg = json.loads((HERE / 'config.json').read_text())
    inputs = json.loads((HERE / 'provenance/inputs.json').read_text())
    assert len(inputs['checkpoints']) == 1
    checkpoint = inputs['checkpoints'][0]
    assert checkpoint['id'] == cfg['condition'] == 'c30'
    weight = next(r for r in checkpoint['files'] if r['path'].endswith('/model.safetensors'))
    assert weight['sha256'] == 'f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425'
    assert cfg['validation_blocks'] == 338 and cfg['excluded_tail_tokens'] == 1444
    assert cfg['timing_inputs'] == 64 and cfg['timing_passes'] == 7 and cfg['process_replicates'] == 3
    assert cfg['modes'] == list(controls.MODES)
    catalog = json.loads((HERE / 'provenance/candidates.json').read_text())
    assert len(catalog['configurations']) == 1
    assert catalog['configurations'][0]['settings']['shortcut'] is False


def test_scientific_matrix(monkeypatch):
    monkeypatch.setitem(sys.modules, 'controls', controls)
    execute = load('run037_execute_test', '03_execute.py')
    assert len(execute.jobs()) == 30
    assert set(execute.jobs()) == {(mode, rep) for mode in controls.MODES for rep in [1, 2, 3]}
    assert len(execute.jobs(True)) == 10
    assert execute.jobs() == execute.jobs()
