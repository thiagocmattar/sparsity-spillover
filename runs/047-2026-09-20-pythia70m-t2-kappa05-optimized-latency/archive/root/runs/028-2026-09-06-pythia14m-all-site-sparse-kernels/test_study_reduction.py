"""Paired aggregation and actual-operand diagnostic audits."""
import copy
import importlib.util
from pathlib import Path
import sys
import pytest
RUN=Path(__file__).parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('run028_reducer',RUN/'65_reduce.py')
reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)


def test_process_range_and_paired_mean_are_not_pooled_median_ratio():
    def process(times):
        return {'samples':[{'mode':mode,'input_index':i,'repeat':0,'host_ms':value}
                           for i,(a,b) in enumerate(times) for mode,value in [('a',a),('b',b)]]}
    timings=[process([(2,1),(20,10)]),process([(3,1),(30,10)]),process([(4,1),(40,10)])]
    value=reducer.paired_reduction(timings,'a','b',inputs=2,passes=1)
    assert value['ratio']==pytest.approx(24**(1/3))
    assert value['process_min']==pytest.approx(2) and value['process_max']==pytest.approx(4)
    assert value['mean_saved_ms']==pytest.approx(11)
    with pytest.raises(ValueError):reducer.paired_reduction(timings,'a','b',inputs=3,passes=1)
    timings[0]['samples'].append(timings[0]['samples'][0])
    with pytest.raises(ValueError):reducer.paired_reduction(timings,'a','b',inputs=2,passes=1)


@pytest.mark.parametrize('condition',['c01','c30'])
def test_completed_smoke_counter_conservation(condition):
    dest=RUN/'artifacts'/f'study-{condition}-smoke-002'
    d=reducer.read_json(dest/'diagnostics.json')
    architecture=reducer.read_json(dest/'result.json')['canonical_logical_products']['architecture_maximum']
    ops=reducer.audit_diagnostics(d,architecture,blocks=8)
    assert len(ops)==6 and ops['mlp_w1']['mma_skipped_fraction']==0
    broken=copy.deepcopy(d);broken['attention_counts_by_layer']['0'][0]+=1
    with pytest.raises(ValueError):reducer.audit_diagnostics(broken,architecture,blocks=8)
    broken=copy.deepcopy(d);broken['active_features_per_row']['a.layer_0'][0]+=1
    with pytest.raises(ValueError):reducer.audit_diagnostics(broken,architecture,blocks=8)
