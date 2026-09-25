"""Routing, interaction, independent scalar work accounting and exactness tests."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from controls import MODES, mask, contrasts, difference_spans
from mechanism_reference import projection_work, same_bits
from qualification import compare_inputs


def brute_work(x, tile, short, fast):
    rows = x.tolist()
    issued = products = groups = 0
    for start in range(0,len(rows),8):
        complex_rows = []
        for row in rows[start:start+8]:
            nonzero = [v for v in row if v != 0]
            eligible = len(nonzero) <= 2 and fast and all(2.**-50 <= abs(v) <= 2.**50 for v in nonzero)
            if short and eligible:
                products += len(nonzero)*128
            else:
                complex_rows.append(row)
        if not complex_rows:
            groups += 1
        else:
            for feature in range(0,len(rows[0]),16):
                if not tile or any(v != 0 for row in complex_rows for v in row[feature:feature+16]):
                    issued += 16
    return issued,products,groups


@pytest.mark.parametrize('tile,short,fast',list(itertools.product([False,True],repeat=3)))
@pytest.mark.parametrize('width',[128,512])
def test_reference_matches_independent_row_and_tile_loop(tile,short,fast,width):
    generator = torch.Generator().manual_seed(71)
    x = torch.zeros(32,width,dtype=torch.bfloat16)
    for row,nnz in enumerate([0,1,2,3,8,0,2,1]*4):
        positions = torch.randperm(width,generator=generator)[:nnz]
        x[row,positions] = torch.randn(nnz,generator=generator).to(torch.bfloat16)
    x[1,0]=2.**-60
    x[9,0]=2.**60
    result = projection_work(x,tile=tile,short=short,fast_weights=fast)['counts']
    issued,products,groups = brute_work(x,tile,short,fast)
    assert result['mma_issued'] == issued
    assert result['scalar_products'] == products
    assert result['short_completed_groups'] == groups
    assert result['mma_issued']+result['mma_omitted_whole_short_group']+result['mma_omitted_empty_tile'] == result['mma_potential']
    assert result['source_weight_requests_bf16_elements'] == issued*128+products


@pytest.mark.parametrize('mode',MODES)
def test_zero_rows_and_bias_only_control_semantics(mode):
    counts = projection_work(torch.zeros(8,128),**mask(mode))['counts']
    s = mask(mode)
    assert counts['scalar_products'] == 0
    assert counts['zero_rows_executed'] == (8 if s['short'] else 0)
    assert counts['mma_omitted_whole_short_group'] == (128 if s['short'] else 0)
    assert counts['mma_omitted_empty_tile'] == (128 if s['tile'] and not s['short'] else 0)


def test_short_routing_enables_additional_empty_tile():
    x = torch.zeros(8,128)
    x[0,:3]=1
    x[1:,17]=1
    rows = {m:projection_work(x,**mask(m))['counts'] for m in MODES}
    assert [rows[m]['mma_issued'] for m in ('t00','t10','t01','t11')] == [128,32,128,16]
    assert rows['t11']['newly_empty_tiles'] == 1
    assert rows['t11']['scalar_products'] == 7*128
    assert rows['t01']['mma_omitted_empty_tile'] == 0


def test_standalone_and_conditional_effects_with_interaction():
    effects = contrasts({'frozen':5.,'t00':10.,'t10':9.,'t01':8.,'t11':5.})
    assert effects == {'tile_given_short_ms':3.,'short_given_tile_ms':4.,'joint_ms':5.,
                       'tile_alone_ms':1.,'short_alone_ms':2.,'interaction_ms':2.}
    assert effects['tile_given_short_ms']+effects['short_given_tile_ms'] == effects['joint_ms']+effects['interaction_ms']
    times={'frozen':5.,'t00':6.,'t10':4.,'t01':3.,'t11':5.}
    assert contrasts(times)['tile_given_short_ms'] == -2.
    with pytest.raises(ValueError): contrasts({**times,'t11':float('nan')})
    ranges={m:[v-.1,v+.1] for m,v in times.items()}
    spans=difference_spans(ranges)
    assert spans['joint_ms'] == pytest.approx([.8,1.2])
    assert spans['interaction_ms'] == pytest.approx([-4.4,-3.6])


def test_signed_zero_is_not_bitwise_equal():
    left=torch.tensor([0.],dtype=torch.bfloat16)
    right=torch.tensor([-0.],dtype=torch.bfloat16)
    assert torch.equal(left,right)
    assert not same_bits(left,right)


def test_numerical_tolerance_does_not_override_frozen_bitwise_gate():
    class Runner:
        def __init__(self,value): self.value=value
        def stage(self,ids): pass
        def __call__(self): return self.value
    value=torch.zeros(1,4,8,dtype=torch.bfloat16)
    other=value.clone();other[0,0,0]=.001
    runners={m:Runner(value) for m in ('native','native_graph','frozen_graph')}
    runners['candidate_graph']=Runner(other)
    bounds={'logit_atol':.25,'logit_rtol':.02,'logit_relative_l2':.02,'validation_loss_atol':.001}
    result=compare_inputs(runners,[torch.zeros(1,4,dtype=torch.long)],bounds,lambda *a,**k:{'pass':True})
    assert result['pass']['candidate_graph']
    assert not result['pass']['candidate_frozen_bitwise']
    assert result['prediction_tokens']==3


def test_input_identity_and_measurement_matrix():
    cfg=json.loads((HERE/'config.json').read_text())
    manifest=json.loads((HERE/'provenance/inputs.json').read_text())
    assert cfg['modes']==list(MODES)
    assert cfg['require_frozen_bitwise']
    assert cfg['validation_blocks']==338 and cfg['excluded_tail_tokens']==1444
    assert cfg['timing_inputs']==64 and cfg['timing_passes']==7 and cfg['process_replicates']==3
    assert len(manifest['checkpoints'])==1 and manifest['checkpoints'][0]['id']=='c30'
    weight=next(r for r in manifest['checkpoints'][0]['files'] if r['path'].endswith('/model.safetensors'))
    assert weight['sha256']=='f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425'
    spec=importlib.util.spec_from_file_location('run056_controller',HERE/'03_execute.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    assert set(controller.jobs())==set(itertools.product(MODES,[1,2,3]))
    assert len(controller.jobs())==15 and len(controller.jobs(True))==5


def test_integer_counts_survive_json_roundtrip(tmp_path):
    from io_utils import read,write,record,verify
    path=tmp_path/'counts.json'
    counts={'numerator':2**60+1,'denominator':2**61+3,'rows':[0,1,2]}
    write(path,counts)
    assert read(path)==counts
    identity=record(path,tmp_path)
    assert verify(identity,tmp_path)==path
    write(path,{**counts,'numerator':counts['numerator']+1})
    with pytest.raises(ValueError):verify(identity,tmp_path)
