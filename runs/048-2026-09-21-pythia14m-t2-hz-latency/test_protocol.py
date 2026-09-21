"""Causal contrast, complete-pair, fixed input, and independent work-count checks."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from controls import MASKS, contrasts


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_conditional_joint_and_interaction():
    effects = contrasts({'A':2., 'B':3., 'C':2.5, 'D':4.})
    assert effects['h_given_z']['saved_ms'] == 1.
    assert effects['z_given_h']['saved_ms'] == .5
    assert effects['joint_hz']['saved_ms'] == 2.
    assert effects['joint_hz']['reduction_percent'] == 50.
    assert effects['interaction_ms'] == .5
    assert contrasts({'A':2., 'B':1.5, 'C':2., 'D':1.})['joint_hz']['saved_ms'] == -1.
    with pytest.raises(ValueError): contrasts({'A':1.})


def test_complete_paired_cells():
    reducer = load('run048_reduce_test', '07_reduce.py')
    samples = [{'repeat':0,'input_index':0,'mode':m,'host_ms':1.,'cuda_ms':.9,
                'output_shape':[1,2048,50304]} for m in reducer.BACKENDS]
    reducer.check_samples(samples, inputs=1, passes=1)
    with pytest.raises(ValueError): reducer.check_samples(samples[:-1], inputs=1, passes=1)
    with pytest.raises(ValueError): reducer.check_samples(samples+[samples[0]], inputs=1, passes=1)
    assert reducer.geomean([1.,4.]) == 2.


def test_frozen_endpoint_and_controls():
    inputs = json.loads((HERE/'provenance/inputs.json').read_text())
    checkpoint, = inputs['checkpoints']
    assert checkpoint['id'] == 'c03' and checkpoint['dose'] == .1
    assert checkpoint['source_condition']['active_sites'] == ['h','z']
    assert checkpoint['source_condition']['pressure_sites'] == ['h']
    assert inputs['validation']['sha256'] == '51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451'
    source = HERE.parent/'037-2026-09-19-pythia14m-operation-latency/candidate/joint.cu'
    assert (HERE/'candidate/joint.cu').read_bytes() == source.read_bytes()
    assert set(MASKS.values()) == {(True,True),(False,True),(True,False),(False,False)}


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('pattern', ['zero','short','mixed','unsafe-small'])
def test_independent_hybrid_counts(monkeypatch, skip, pattern):
    monkeypatch.setitem(sys.modules, 'common', SimpleNamespace(write_json=None))
    diagnostic = load('run048_diagnostic_test', 'diagnostics.py')
    value = torch.zeros(16,512)
    if pattern != 'zero': value[:,0]=.75; value[:,17]=-.5
    if pattern == 'mixed': value[8:,32:48]=.625
    if pattern == 'unsafe-small': value[:,0]=2.**-60
    expected = {'zero':[0,1024,0],'short':[0,1024,4096],
                'mixed':[48,976,2048],'unsafe-small':[64,960,0]}[pattern]
    assert diagnostic.hybrid_counts(value, True, skip) == (expected if skip else [1024,0,0])
