"""Mathematical and evidence-contract checks for the results package."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import HERE, ROOT, OPS, SCALES, DOSES, frontier, load_evidence, logical_counts, mass_bands, one, sha


@pytest.fixture(scope='module')
def data():
    return load_evidence()


def test_frontier_keeps_tradeoffs_and_ties_but_removes_dominated():
    rows=[{'id':'a','loss':1.,'R_model':.1},{'id':'tie','loss':1.,'R_model':.1},
          {'id':'b','loss':2.,'R_model':.3},{'id':'bad','loss':2.,'R_model':.2},
          {'id':'worse','loss':3.,'R_model':.3}]
    assert {r['id'] for r in frontier(rows)} == {'a','tie','b'}


def test_mass_bands_partition_ties_and_reject_nonmonotone_counts():
    row={'total':100,'exact_zero_count':20,'threshold_hits':{'0.001':20,'0.01':65}}
    assert mass_bands(row)==[20,0,45,35]
    row['threshold_hits']['0.001']=19
    with pytest.raises(AssertionError): mass_bands(row)


def test_complete_condition_and_clipping_grids(data):
    assert [sum(r['scale']==s for r in data['trained']) for s in SCALES]==[35,12,12]
    assert [sum(r['scale']==s for r in data['clipping']) for s in SCALES]==[150,20,20]
    assert len({r['id'] for r in data['trained']+data['clipping']})==249
    assert len(data['contrasts'])==30
    for scale in SCALES:
        for family in ('A4-OL1','A7-OL1'):
            assert sorted(r['dose'] for r in data['trained'] if r['scale']==scale and r['family']==family)==list(DOSES)


def test_counts_reconcile_and_corruption_is_rejected(data):
    for r in data['trained']+data['clipping']:
        logical_counts(r['counts'])
    bad=copy.deepcopy(data['trained'][0]['counts'])
    bad['per_operation'][OPS[0]]['zero_product_count']+=1
    with pytest.raises(AssertionError): logical_counts(bad)


def test_normalizations_preserve_natural_zeros_and_undefined_a0(data):
    for r in data['trained']:
        c,a=r['counts'],r['ceiling'];reach=338*a['reachable_product_count']
        assert c['model_product_count']==338*a['model_product_count']
        if r['family']=='A0':
            assert r['U_arch'] is None and r['U_reach'] is None
            assert r['R_model']>0
        else:
            assert r['U_arch']==c['block_zero_product_count']/reach
            assert r['U_reach']==(c['block_zero_product_count']-r['outside_reach_zero_products'])/reach
            assert 0<=r['U_reach']<=1
            assert r['U_arch']>=r['U_reach']
        if r['family'].startswith('A7'):
            assert r['outside_reach_zero_products']==0
            assert r['U_arch']==c['block_zero_product_count']/c['block_product_count']
def test_operation_weighting_reconstructs_sparsity(data):
    for r in data['trained']:
        c=r['counts']
        reconstructed=sum((o['zero_product_count']/o['product_count'])*
                          (o['product_count']/c['model_product_count']) for o in c['per_operation'].values())
        assert reconstructed==pytest.approx(r['R_model'],abs=1e-12)


def test_contrasts_have_matched_identities_and_correct_signs(data):
    for pair in data['contrasts']:
        a=one(data['trained'],id=pair['parent']);b=one(data['trained'],id=pair['child'])
        assert a['scale']==b['scale']=='14M'
        for key in ('initial_parameter_sha256','schedule_sha256','validation_sha256','training_tokens','seeds'):
            assert a['identity'][key]==b['identity'][key]
        assert pair['delta_loss']==b['loss']-a['loss']
        assert pair['delta_R_pp']==100*(b['R_model']-a['R_model'])


def test_case_study_uses_common_sites_and_pooled_band_counts(data):
    common={'m','h','q_post','k_post','v','attention_output'}
    selected=[one(data['trained'],scale='14M',family='A0')]
    selected += [one(data['trained'],scale='14M',family=f,dose=k) for f in ('A4-OL1','A7-OL1') for k in (0.,.5)]
    assert set(selected[0]['sites'])==common
    for r in selected:
        for site in common:
            a=r['sites'][site]
            assert sum(a['mass_bands'])==a['total']
            assert a['mass_bands'][0]==a['exact_zero_count']
            assert a['rms']==pytest.approx(math.sqrt(a['square_sum']/a['finite']))


def test_historical_pressure_cannot_enter_main_ladder_frontier(data):
    historical=[r for r in data['trained'] if r['family']=='A4-OL1[h]']
    assert len(historical)==5 and all(r['source'].startswith('runs/012-') for r in historical)
    corrected=[r for r in data['trained'] if r['scale']=='14M' and r['family']=='A4-OL1']
    assert len(corrected)==5 and all(r['source'].startswith('runs/015-') for r in corrected)
    assert not any('OL1[h]' in id_ for id_ in data['frontiers']['14M_main_trained'])


def test_scale_transfer_claim_and_normalization_order(data):
    for scale in SCALES:
        high=one(data['scale_pairs'],scale=scale,dose=.5)
        assert high['delta_loss']<0 and high['delta_R_pp']>0
        low=one(data['scale_pairs'],scale=scale,dose=0.)
        if scale=='410M': assert low['delta_loss']<0 and low['delta_R_pp']>0
        else: assert low['delta_loss']>0 and low['delta_R_pp']<0
    assert all(r['delta_U_pp']<0 for r in data['scale_pairs'])


def test_saved_bundle_and_all_sources_match(data):
    assert json.loads((HERE/'figure_data.json').read_text(encoding='utf-8'))==data
    for path,digest in data['sources'].items():
        assert sha(ROOT/path)==digest
    inventory=json.loads((HERE/'artifact_inventory.json').read_text())
    assert len(list((HERE/'figures').glob('*.pdf')))==7
    for path,meta in inventory.items():
        assert sha(HERE/path)==meta['sha256'] and (HERE/path).stat().st_size==meta['bytes']
    assert sha(HERE/'figures/07-kernel-realization.pdf')==sha(ROOT/data['kernel_pdf_source'])
