"""Mathematical and evidence-contract checks for the results package."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import HERE, ROOT, OPS, SCALES, DOSES, CASE_DOSES, CASE_SITES, frontier, load_evidence, logical_counts, mass_bands, one, sha


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
    assert [sum(r['scale']==s for r in data['trained']) for s in SCALES]==[30,12,12]
    assert [sum(r['scale']==s for r in data['clipping']) for s in SCALES]==[150,20,20]
    assert len({r['id'] for r in data['trained']+data['clipping']})==244
    assert len(data['contrasts'])==29
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
        a=one(data['trained'],id=pair['reference']);b=one(data['trained'],id=pair['treatment'])
        assert a['scale']==b['scale']=='14M'
        for key in ('initial_parameter_sha256','schedule_sha256','validation_sha256','training_tokens','seeds'):
            assert a['identity'][key]==b['identity'][key]
        assert pair['delta_loss']==b['loss']-a['loss']
        assert pair['delta_R_pp']==100*(b['R_model']-a['R_model'])


def test_a1h_to_a4_covers_all_thresholds_with_fixed_reference(data):
    rows=[r for r in data['contrasts'] if r['block']=='A1-H to A4']
    reference=one(data['trained'],scale='14M',family='A1-H')
    assert [r['dose'] for r in rows]==list(DOSES)
    for r in rows:
        assert r['reference']==reference['id']
        treatment=one(data['trained'],scale='14M',family='A4',dose=r['dose'])
        assert r['treatment']==treatment['id']


def test_case_study_uses_common_sites_and_pooled_band_counts(data):
    common={'m','h','q_post','k_post','v','attention_output'}
    selected=[one(data['trained'],scale='14M',family='A0')]
    selected += [one(data['trained'],scale='14M',family=f,dose=k) for f in ('A4-OL1','A7-OL1') for k in CASE_DOSES]
    assert len(selected)==7 and list(CASE_DOSES)==data['activation_case']['kappas']
    assert set(selected[0]['sites'])==common
    for r in selected:
        for site in common:
            a=r['sites'][site]
            assert sum(a['mass_bands'])==a['total']
            assert a['mass_bands'][0]==a['exact_zero_count']
            assert a['rms']==pytest.approx(math.sqrt(a['square_sum']/a['finite']))


def test_overview_retains_complete_dose_sweeps_in_order(data, monkeypatch):
    # Assert the actual plotted coordinates, not just a reduction agreeing with itself.
    import plots
    figures=[]
    monkeypatch.setattr(plots, 'save', lambda fig, name: figures.append(fig))
    plots.configure()
    plots.overview(data)
    expected={'A0': [None], 'A1-H': [None],
              'A1-H-L1': [.05,.1,.5,1.], 'A1-H-OL1': [.05,.1,.5,1.],
              **{f: [0.,.01,.05,.1,.5] for f in ('A4','A4-OL1','A7','A7-OL1')}}
    sources={'A0 + clipping':'gelu-control','A1-H + clipping':'relu-control'}
    raw_sources=[f'relu-{method}-{level}' for method in ('l1n','ol1')
                 for level in ('0p05','0p1','0p5','1')]
    raw_sources += [f'a4z-one-sided-kappa-{level}' for level in ('0','0p01','0p05','0p1','0p5')]
    sources.update({source+' + clipping':source for source in raw_sources})
    expected.update({label:[i/10 for i in range(10)] for label in sources})
    try:
        lines=figures[0].axes[0].get_lines()
        assert len(lines)==23
        assert {line.get_label() for line in lines}==set(expected)
        assert set(data['overview_series'])==set(expected)
        assert sum(len(line.get_xdata()) for line in lines)==180
        for line in lines:
            series=line.get_label()
            clipping=series.endswith(' + clipping')
            family=sources[series] if clipping else series
            candidates=[r for r in data['clipping' if clipping else 'trained']
                        if r['scale']=='14M' and r['family']==family]
            rows=[one(candidates,dose=dose) for dose in expected[series]]
            assert data['overview_series'][series]==[r['id'] for r in rows]
            assert list(line.get_xdata())==[100*r['R_model'] for r in rows]
            assert list(line.get_ydata())==[r['loss'] for r in rows]
        # The actual measured p=0 values stay in the thin trajectories, while
        # canonical trained markers render above them instead of being hidden.
        clipped=[line for line in lines if line.get_label() in sources]
        trained=[line for line in lines if line.get_label() not in sources]
        assert min(line.get_zorder() for line in trained)>max(line.get_zorder() for line in clipped)
        # This low-dose point was previously omitted because a higher dose dominates it.
        a4=[r for r in data['trained'] if r['scale']=='14M' and r['family']=='A4']
        assert one(a4,dose=0.) not in frontier(a4)
    finally:
        for fig in figures: plots.plt.close(fig)


def test_historical_pressure_is_excluded_from_all_current_results(data):
    historical=[r for r in data['trained'] if r['family']=='A4-OL1[h]']
    assert not historical
    assert all(p['family'] != 'A4+OL1@h' and int(p['condition'][1:]) <= 30 for p in data['runtime']['points'])
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
    assert len(list((HERE/'figures').glob('*.pdf')))==9
    for path,meta in inventory.items():
        assert sha(HERE/path)==meta['sha256'] and (HERE/path).stat().st_size==meta['bytes']


def test_clipping_uses_evaluation_site_reach_even_for_a0_at_p_zero(data):
    for row in data['clipping']:
        ceiling=one(data['ceilings'],scale=row['scale'],family='A4')
        assert set(row['normalization_sites'])=={'a','m','h','z'}
        assert row['ceiling']['reachable_product_count']==ceiling['reachable_product_count']
        assert row['U_arch']==pytest.approx(row['R_model']/ceiling['R_model_max_fraction'])
    assert sum(r['control'] is not None for r in data['clipping'])==60


def test_ceiling_grid_has_integer_units_and_shared_ol1_topologies(data):
    assert len(data['ceilings'])==12
    for scale in SCALES:
        for f in ('A4','A7'):
            ceiling=one(data['ceilings'],scale=scale,family=f)
            assert isinstance(ceiling['reachable_product_count'],int)
            assert '2048-token sequence' in ceiling['unit']
            assert ceiling['R_model_max_fraction']==pytest.approx(
                ceiling['reachable_product_count']/ceiling['model_product_count'])
            assert ceiling['R_model_max_fraction']==one(data['trained'],scale=scale,family=f+'-OL1',dose=0.)['ceiling']['R_model_max_fraction']


def test_runtime_subset_reduction_uses_thirty_qualified_checkpoints(data):
    runtime=data['runtime'];rows=[r for r in runtime['points'] if r['candidate']=='k050']
    assert len(rows)==30 and all(r['qualified'] for r in rows)
    assert runtime['final_candidates']['k050']['geomean']==pytest.approx(
        math.prod(r['speedup'] for r in rows)**(1/30))
    fit=runtime['k050_regression'];mean=sum(r['speedup'] for r in rows)/30
    sse=sum((r['speedup']-fit['intercept']-fit['slope_per_fraction']*r['R_model'])**2 for r in rows)
    assert fit['r2']==pytest.approx(1-sse/sum((r['speedup']-mean)**2 for r in rows))
