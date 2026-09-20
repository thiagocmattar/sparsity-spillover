import copy
import importlib.util
from pathlib import Path
import sys
import pytest

RUN=Path(__file__).parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('hybrid_reducer_test',RUN/'121_reduce_hybrid.py')
reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)


def fixture():
    folder=RUN/'artifacts/study-c30-smoke-002'
    d=reducer.read_json(folder/'diagnostics.json')
    architecture=reducer.read_json(folder/'result.json')['canonical_logical_products']['architecture_maximum']
    # Synthetic dense padded counters atop a real, fully covered site fixture.
    d['hybrid_counts_by_layer']={str(i):[131072*8,0,32768*8,0,0,0] for i in range(6)}
    for op,total in [('mlp_w2',131072*8*6),('attention_output_projection',32768*8*6)]:
        d['bf16_scalar_opportunity_lower_bound']['per_operation'][op].update(issued_mmas=total,skipped_mmas=0,simt_products=0)
    return d,architecture


def test_hybrid_padding_and_per_layer_pooling():
    d,a=fixture();ops=reducer.audit_diagnostics(d,a,blocks=8)
    assert ops['mlp_w2']['mma_total']*2048==2*ops['mlp_w2']['products']
    assert ops['mlp_w1']['mma_total']*2048==ops['mlp_w1']['products']
    assert ops['mlp_w2']['simt_products']==0
    for field in ['issued_mmas','simt_products']:
        broken=copy.deepcopy(d);broken['bf16_scalar_opportunity_lower_bound']['per_operation']['mlp_w2'][field]+=1
        with pytest.raises(ValueError):reducer.audit_diagnostics(broken,a,blocks=8)
    broken=copy.deepcopy(d);broken['hybrid_counts_by_layer']['0'][1]+=1
    with pytest.raises(ValueError):reducer.audit_diagnostics(broken,a,blocks=8)


def test_nine_modes_and_separate_fusion_comparison():
    assert len(reducer.MODES)==9 and len(set(reducer.MODES))==9
    assert reducer.COMPARISONS['fusion']==('unfused_graph','sparse_graph')
    assert reducer.COMPARISONS['skip']==('no_skip_graph','sparse_graph')


@pytest.mark.parametrize('condition',['c01','c30'])
def test_actual_hybrid_smoke_counter_and_numerical_coverage(condition):
    folder=RUN/'artifacts'/f'hybrid-{condition}-smoke-001'
    result=reducer.read_json(folder/'result.json')
    quality=reducer.read_json(folder/'quality.json')
    timing=reducer.read_json(folder/'timing.json')
    assert result['status']=='complete' and quality['blocks']==8
    assert set(quality['pass'])==set(reducer.MODES) and all(quality['pass'].values())
    assert len(timing['samples'])==9*4*2
    d=reducer.read_json(folder/'diagnostics.json')
    ops=reducer.audit_diagnostics(d,result['canonical_logical_products']['architecture_maximum'],blocks=8)
    assert len(ops)==6
    if condition=='c30':
        assert ops['mlp_w2']['simt_products']>0 and ops['attention_output_projection']['simt_products']>0
