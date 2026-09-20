"""Scientific cohort, source integrity and estimand/coverage regression checks."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import pytest

RUN = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('run047_'+name, RUN/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_matrix_and_matching_protocol():
    config = json.loads((RUN/'config.json').read_text())
    matrix = load('03_execute').jobs(config, 'final')
    assert len(matrix) == len(set(matrix)) == 9
    assert set(matrix) == {(c, r) for c in ('c00','c25','c26') for r in range(1, 4)}
    assert (config['timing_inputs'], config['timing_passes']) == (64, 7)
    assert config['numerical_bounds'] == dict(logit_atol=.25, logit_rtol=.02,
        logit_relative_l2=.02, validation_loss_atol=.001)


def test_reference_and_new_checkpoint_identities():
    data = json.loads((RUN/'provenance/inputs.json').read_text())
    rows = data['checkpoints']
    assert [r['id'] for r in rows] == ['c00','c25','c26']
    assert [r['family'] for r in rows] == ['A0','HZ+OL1@h','HZ+OL1@h']
    assert [r['dose'] for r in rows[1:]] == [.1,.5]
    assert rows[2]['final_checkpoint_content_sha256']=='d6a3813f83ce080b07637fe422bb7ab1b23a6793e39c365601301a661851d02f'
    previous = json.loads((RUN.parents[1]/'runs/045-2026-09-20-pythia70m-kernel-grid/provenance/inputs.json').read_text())
    for row in rows[:2]:
        reference = next(r for r in previous['checkpoints'] if r['id']==row['id'])
        assert row['final_checkpoint_content_sha256']==reference['final_checkpoint_content_sha256']
        assert row['canonical_logical_products']==reference['canonical_logical_products']
    assert data['validation']['sha256']=='51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451'
    for row in rows:
        assert row['canonical_logical_products']['coverage']['sequences']==338
        assert row['canonical_logical_products']['coverage']['excluded_tail_tokens']==1444
        if row['family']=='HZ+OL1@h':
            assert row['canonical_logical_products']['architecture_maximum']['active_sites']==['h','z']


def test_bridge_only_extends_topology_acceptance():
    original = (RUN/'archive/root/runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/adapter.py').read_text()
    changed = original.replace("{'A0','A1-H','A4-Z','A7-Z-POST'}", "{'A0','A1-H','A4-Z','A7-Z-POST','HZ'}")
    assert changed != original and (RUN/'hz_adapter.py').read_text() == changed
    source = RUN.parents[1]/'runs/045-2026-09-20-pythia70m-kernel-grid'
    for folder in ('kernel','base70','candidates'):
        for path in (RUN/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                assert path.read_bytes()==(source/path.relative_to(RUN)).read_bytes()
    for name in ('02_benchmark.py','diagnostics.py','replay.py','controls.py','frozen_replay.py'):
        assert (RUN/name).read_bytes()==(source/name).read_bytes()


def test_geometric_mean_is_not_median_or_arithmetic_mean():
    reduce = load('06_reduce')
    assert reduce.geomean([1., 1., 64.]) == pytest.approx(4.)
    assert reduce.geomean([.1, 10.]) == pytest.approx(1.)
    for values in ([], [0.], [-1.], [math.nan], [math.inf]):
        with pytest.raises(ValueError):
            reduce.geomean(values)


def test_paired_cells_reject_duplicates_missing_modes_and_partial_logits():
    reduce = load('06_reduce')
    timing = {'indices': [27, 305], 'samples': [
        {'mode': mode, 'input_index': i, 'repeat': r, 'host_ms': 1., 'output_shape': [1, 2048, 50304]}
        for mode in reduce.MODES for i in range(2) for r in range(3)]}
    assert all(len(v) == 6 for v in reduce.audit_samples(timing, 2, 3).values())
    duplicate = copy.deepcopy(timing)
    duplicate['samples'][0] = duplicate['samples'][1]
    incomplete = copy.deepcopy(timing)
    incomplete['samples'].pop()
    truncated = copy.deepcopy(timing)
    truncated['samples'][0]['output_shape'] = [1, 1, 50304]
    for bad in (duplicate, incomplete, truncated):
        with pytest.raises(ValueError):
            reduce.audit_samples(bad, 2, 3)


def test_quality_requires_each_block_and_pooled_prediction_tokens():
    reduce = load('06_reduce')
    bounds = json.loads((RUN/'config.json').read_text())['numerical_bounds']
    quality = dict(blocks=338, documents=500, input_tokens=692224, prediction_tokens=691886,
                   excluded_tail_tokens=1444, loss={'native': 4.}, gates={}, **{'pass': {}})
    for mode in reduce.MODES:
        quality['loss'][mode] = 4.
        quality['gates'][mode] = [dict(input_index=i, finite=True, elementwise_gate=True,
                                      relative_l2=0.) for i in range(338)]
        quality['pass'][mode] = True
    assert all(reduce.audit_quality(quality, bounds).values())
    failed = copy.deepcopy(quality)
    failed['gates']['candidate_graph'][17]['elementwise_gate'] = False
    failed['pass']['candidate_graph'] = False
    assert reduce.audit_quality(failed, bounds)['candidate_graph'] is False
    failed['pass']['candidate_graph'] = True
    with pytest.raises(ValueError):
        reduce.audit_quality(failed, bounds)
    for key, value in [('prediction_tokens', 692224), ('blocks', 337)]:
        bad = copy.deepcopy(quality)
        bad[key] = value
        with pytest.raises(ValueError):
            reduce.audit_quality(bad, bounds)
    bad = copy.deepcopy(quality)
    bad['gates']['native_graph'][1]['input_index'] = 0
    with pytest.raises(ValueError):
        reduce.audit_quality(bad, bounds)


def test_reducer_rejects_incomplete_followup(tmp_path, monkeypatch):
    reduce=load('06_reduce')
    config=json.loads((RUN/'config.json').read_text())
    (tmp_path/'provenance').mkdir()
    (tmp_path/'artifacts/final').mkdir(parents=True)
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'provenance/inputs.json').write_text((RUN/'provenance/inputs.json').read_text())
    incomplete=[dict(condition=c,replicate=r) for c in config['conditions'] for r in (1,2,3)][:-1]
    (tmp_path/'artifacts/final/summary-test.json').write_text(json.dumps(incomplete))
    monkeypatch.setattr(reduce,'RUN',tmp_path)
    with pytest.raises(ValueError,match='nine-process'):
        reduce.reduce('test')
