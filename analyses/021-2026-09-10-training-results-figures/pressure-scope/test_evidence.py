"""Checks for changed cohorts, paired estimands and historical timing reuse."""
import json
from pathlib import Path

import numpy as np
import pytest

import evidence
import figures


@pytest.fixture(scope="module")
def data():
    rebuilt=evidence.load()
    assert rebuilt == json.loads((evidence.HERE/'data/evidence.json').read_text())
    return rebuilt


def test_cohort_identity_and_integer_pooling(data):
    assert data['training_endpoint_count']==64
    assert len(data['rows'])==30
    for family in evidence.FAMILIES:
        selected=[r for r in data['rows'] if r['family']==family]
        assert {r['kappa'] for r in selected}==set(evidence.KAPPAS)
        for r in selected:
            expected=['h'] if family.endswith('-H') else r['gate_sites'] if 'OL1' in family else []
            assert r['realized_pressure_sites']==expected
            for site in evidence.SITES:
                layers=r['layers'][site];pool=r['sites'][site]
                assert len(layers)==6
                assert pool['zero']==sum(p['zero'] for p in layers)
                assert pool['total']==sum(p['total'] for p in layers)
                assert pool['percent']==pytest.approx(100*pool['zero']/pool['total'])
            assert sum(r['operations_pp'].values())==pytest.approx(r['S_model_percent'])


def test_matched_contrasts_and_loss_pass_sensitivity(data):
    assert len(data['contrasts'])==30
    for c in data['contrasts']:
        a=next(r for r in data['rows'] if r['family']==c['treatment'] and r['kappa']==c['kappa'])
        b=next(r for r in data['rows'] if r['family']==c['reference'] and r['kappa']==c['kappa'])
        assert c['delta_loss']==a['loss']-b['loss']
        assert c['delta_s_pp']==a['S_model_percent']-b['S_model_percent']
        assert abs(c['delta_loss']-c['delta_loss_logical']) <= 2*data['max_loss_pass_difference']
    c=next(c for c in data['contrasts'] if c['treatment']=='A7-OL1-H' and c['reference']=='A4-OL1-H' and c['kappa']==.5)
    assert c['delta_s_pp']==pytest.approx(6.436253037915803)
    assert c['delta_loss']==pytest.approx(.009383100024341)


def test_geometry_uses_all_steps_and_implemented_cap(data):
    assert sum(len(r['steps']) for r in data['geometry'])==14240
    for f,g in data['geometry_groups'].items():
        steps=[s for r in data['geometry'] if r['family']==f for s in r['steps']]
        assert len(steps)==3560
        assert g['cap_percent']==100*sum(s['cap'] for s in steps)/len(steps)
        assert g['median_rho']==np.median([s['rho'] for s in steps])
    assert data['geometry_groups']['A4-OL1-H']['cap_percent']==pytest.approx(100*9/3560)
    assert data['geometry_groups']['A7-OL1-H']['cap_percent']==pytest.approx(100*9/3560)


def test_runtime_restores_five_qualified_h_only_records(data):
    allrows=data['runtime']+data['h_only_runtime']
    assert {r['condition'] for r in allrows}=={f'c{i:02d}' for i in range(1,36)}
    assert all(r['attention_sparse_gain']<1 for r in allrows)
    for r in data['h_only_runtime']:
        assert r['qualified'] and r['replicates']==3
        assert r['projection_sparse_gain']==pytest.approx(r['all_skips_off_candidate_gm_ms']/r['projection_on_candidate_gm_ms'])
        assert r['full_speedup']==pytest.approx(r['native_baseline_gm_ms']/r['full_candidate_gm_ms'])
    assert data['runtime_fits_35']['projection_mma_bypass_fraction']==pytest.approx(.9414663044715486)
    fig=figures.kernel(data)
    left,right=fig.axes
    # The native A0 cross is a reference, not a 36th optimized checkpoint.
    assert sum(len(c.get_offsets()) for c in left.collections)==36
    assert sum(len(c.get_offsets()) for c in right.collections)==35
    offsets=np.concatenate([c.get_offsets() for c in left.collections[:-1]])
    np.testing.assert_allclose(sorted(map(tuple,offsets)),sorted((r['full_candidate_gm_ms'],r['validation_loss']) for r in allrows))
    figures.plt.close(fig)
