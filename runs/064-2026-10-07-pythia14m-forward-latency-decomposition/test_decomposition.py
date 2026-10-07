import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('run064_accounting',Path(__file__).with_name('accounting.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_overlap_is_not_double_counted():
    assert a.intervals_union([(1,4),(2,3),(3,6),(8,9)])==6

def test_stage_budget_closes_by_arithmetic_pooling():
    result=a.budget(['a','b'],[[1,9],[3,7]])
    assert result['total_us']==10
    assert sum(row['fraction'] for row in result['stages'])==1
    assert result['stages'][0]['mean_us']==2

@pytest.mark.parametrize('samples',[[[1]],[[1,float('nan')]],[[1,-1]]])
def test_reject_incomplete_or_nonfinite(samples):
    with pytest.raises(ValueError):a.budget(['a','b'],samples)
