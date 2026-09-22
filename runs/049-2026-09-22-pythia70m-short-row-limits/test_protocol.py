"""Focused checks for training-only selection and exact mixed-group accounting."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))


def load(name):
    spec = importlib.util.spec_from_file_location('run049_test_' + name, RUN / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_selection_requires_all_conditions_and_uses_primary_latency():
    execute = load('03_execute')
    rows = []
    for cid in ('c00', 'c24', 'c25'):
        rows.append({'arguments': {'condition': cid},
            'qualification': {m + '_graph': True for m in ('native_hz', 'limit16', 'limit32', 'limit64')},
            'timing': {m + '_graph': {'geomean_host_ms': t} for m, t in
                      [('native_hz', 1.3), ('limit16', 1.2), ('limit32', 1.1), ('limit64', 1.0)]}})
    rows[0]['qualification']['limit64_graph'] = False
    rows[1]['qualification']['limit32_graph'] = False
    assert execute.choose(rows, {'primary_condition': 'c25'}) == 'limit16'
    rows[2]['timing']['limit16_graph']['geomean_host_ms'] = 1.4
    assert execute.choose(rows, {'primary_condition': 'c25'}) == 'native_hz'


def test_geomean_pools_observations_and_rejects_invalid_timing():
    reduce = load('06_reduce')
    assert reduce.geomean([1., 4., 16.]) == pytest.approx(4.)
    for values in ([], [0.], [-1.], [float('nan')], [float('inf')]):
        with pytest.raises(ValueError):
            reduce.geomean(values)


@pytest.mark.parametrize('limit', [8, 16, 32, 64])
def test_capacity_boundary_and_mixed_group_duplicate_oracle(limit):
    count = load('tile_oracle').counts
    extra = load('parallel_work_oracle').extra_scalar_products
    h = torch.zeros(8, 2048)
    z = torch.zeros(8, 512)
    h[:, :2] = 1
    h[0, :limit+1] = 1
    issued, skipped, scalar = count(h, 8, True, True, limit)
    active_tiles = (limit + 1 + 15) // 16
    assert issued == active_tiles * 64
    assert issued + skipped == 128 * 64
    assert scalar == 7 * 2 * 512
    assert extra(h, z, limit=limit) == [7 * 2 * 512, 0]
    h[0, limit] = 0
    assert count(h, 8, True, True, limit) == [0, 128*64, (limit+14)*512]
    assert extra(h, z, limit=limit) == [0, 0]


def test_retained_data_and_checkpoint_coverage():
    io = load('io_utils')
    cfg = io.read(RUN / 'config.json')
    inputs = io.read(RUN / 'provenance/inputs.json')
    assert [r['id'] for r in inputs['checkpoints']] == ['c00', 'c24', 'c25']
    assert [(r['family'], r['dose']) for r in inputs['checkpoints']] == [('A0', None), ('HZ+OL1@h', .05), ('HZ+OL1@h', .1)]
    assert inputs['validation']['sha256'] == '51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451'
    assert cfg['validation_blocks'] == 338 and cfg['excluded_tail_tokens'] == 1444
    assert cfg['timing_inputs'] * cfg['timing_passes'] * cfg['process_replicates'] == 1344
