import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import HERE, load_evidence, paired_delta, nondominated, one, read


@pytest.fixture(scope='module')
def evidence():
    return load_evidence()


def test_count_and_loss_contract(evidence):
    assert len(evidence['trained']) == 59
    assert len(evidence['clipping']) == 60
    for r in evidence['trained']:
        if 'counts' in r:
            c=r['counts']
            assert r['R_model'] == c['block_zero_product_count']/c['model_product_count']
            assert abs(r['loss']-r['terminal_loss']) < .001


def test_paired_sign_and_units():
    a={'scale':'x','dose':.5,'loss':5.,'R_model':.1}
    b={**a,'loss':5.2,'R_model':.22}
    d=paired_delta(a,b)
    assert d['delta_loss'] == pytest.approx(.2)
    assert d['delta_R_pp'] == pytest.approx(12.)
    with pytest.raises(ValueError):
        paired_delta(a,{**b,'scale':'other'})


def test_pareto_directions_and_ties():
    rows=[{'loss':1.,'R_model':.2}, {'loss':2.,'R_model':.3},
          {'loss':2.,'R_model':.1}, {'loss':1.,'R_model':.2}]
    assert nondominated(rows) == [rows[0], rows[1], rows[3]]


def test_scope_and_pressure(evidence):
    h=[r for r in evidence['trained'] if r['family']=='A4+OL1@h']
    assert len(h)==5 and all('/012-' in r['source'] for r in h)
    for r in evidence['contrasts']:
        if r['parent']=='A4+OL1@h':
            assert r['delta_loss'] > .24
    high=one(evidence['pressure_response_differences'],dose=.5)
    assert high['delta_R_pp'] == pytest.approx(9.59796,abs=.00001)
    assert high['delta_loss'] < -.25


def test_model_share_explains_size_comparison(evidence):
    a=one(evidence['trained'],scale='14M',family='A7+OL1@7',dose=.5)
    b=one(evidence['trained'],scale='70M',family='A7+OL1@7',dose=.5)
    assert a['R_model'] < b['R_model'] and a['R_block'] > b['R_block']
    for r in (a,b):
        c=r['counts']
        assert r['R_model'] == pytest.approx(r['R_block']*c['block_product_count']/c['model_product_count'])


def test_runtime_scope_and_adverse_evidence(evidence):
    assert evidence['runtime_verification']['passed']
    assert len(evidence['runtime'])==6
    assert all(r['b1_speedup_p90'] < 1 for r in evidence['runtime'])
    a=one(evidence['runtime'],condition_id='a7-ol1-kappa-0p5')
    assert a['R_covered_linear'] < a['R_model']
    assert a['attention_composition_speedup_max'] < 1
    assert all(r['b32_paired_speedup'] < 1 for r in evidence['runtime'])


def test_complete_clipping_grid(evidence):
    for scale in ('14M', '70M', '410M'):
        for family in ('A0 clipped', 'A1-H clipped'):
            doses = sorted(r['dose'] for r in evidence['clipping']
                           if r['scale'] == scale and r['family'] == family)
            assert doses == [i/10 for i in range(10)]


def test_serialized_common_loss_selections(evidence):
    bundle = read(HERE/'figure_data.json')
    assert len(bundle['loss_budget_comparisons']) == 20
    for row in bundle['loss_budget_comparisons']:
        base = one(evidence['clipping'], scale=row['scale'],
                   family='A0 clipped', dose=0.)['loss']
        eligible = [r for r in evidence[row['kind']] if r['scale'] == row['scale']
                    and r['loss'] <= base+row['budget']]
        assert row['selected']['R_model'] == max(r['R_model'] for r in eligible)
    def selected(scale, budget, kind):
        return one(bundle['loss_budget_comparisons'], scale=scale, budget=budget,
                   kind=kind)['selected']['R_model']
    assert selected('14M',.05,'trained') > selected('14M',.05,'clipping')
    assert selected('70M',.25,'trained') < selected('70M',.25,'clipping')
    assert selected('70M',1.,'trained') > selected('70M',1.,'clipping')


def test_410m_reversal_and_training_only_lr_selection(evidence):
    for family in ('A4+OL1@4', 'A7+OL1@7'):
        low = one(evidence['trained'],scale='410M',family=family,dose=0.)
        high = one(evidence['trained'],scale='410M',family=family,dose=.5)
        assert high['loss'] < low['loss'] and high['R_model'] > low['R_model']
    screen = evidence['lr_screen']
    assert screen['selection_uses_validation'] is False
    best = min(screen['arms'], key=lambda r:r['selection_metric'])
    assert best['condition_id'] == screen['selected_condition_id'] == 'a0-gelu'


def test_high_threshold_branch_suppression_and_assets(evidence):
    for scale in ('14M','70M'):
        for family in ('A4+OL1@4','A7+OL1@7'):
            row=one(evidence['trained'],scale=scale,family=family,dose=.5)
            assert all(row['sites'][s]['fraction'] > .998 for s in ('h','z'))
    assert len(list((HERE/'figures').glob('*.pdf'))) == 5
    assert len(list((HERE/'tables').glob('*.tex'))) == 12
    for figure in (HERE/'figures').glob('*.pdf'):
        assert figure.read_bytes().startswith(b'%PDF-')
