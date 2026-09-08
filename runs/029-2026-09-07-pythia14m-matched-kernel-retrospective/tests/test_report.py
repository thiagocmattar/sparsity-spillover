from pathlib import Path
import sys
import pytest

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
from io_utils import module
report = module('run029_report_test', RUN/'16_report.py')


def test_describe_uses_geometric_mean():
    assert report.describe([.5, 2.]) == {'n': 2, 'geomean': 1., 'minimum': .5, 'maximum': 2., 'above_one': 1}
    assert report.describe([]) == {'n': 0}
    with pytest.raises(ValueError):
        report.describe([0.])


def test_controls_are_checkpoint_matched_and_jointly_qualified():
    points = [{'phase': 'final', 'candidate': c, 'condition': condition,
               'R_model': .3, 'speedup': speed, 'qualified': not (c == 'p0' and condition == 'c02')}
              for condition in ['c01', 'c02']
              for c, speed in [('k050', 2.), ('p0', .5), ('k050-no-skip', 1.25),
                               ('k050-attention-dense', 2.2), ('k049', 1.9)]]
    result = report.comparisons(points)
    assert result['summary']['p0']['n'] == 1
    assert result['summary']['p0']['geomean'] == 4.
    assert result['summary']['k050-no-skip']['geomean'] == pytest.approx(1.6)
    assert result['summary']['k050-attention-dense']['above_one'] == 0


def test_numerical_report_separates_elementwise_and_loss_failures():
    quality = {'gates': {'candidate_graph': [
        {'pass': False, 'finite': True, 'elementwise_gate': False, 'relative_l2': .001, 'max_abs': .75},
        {'pass': True, 'finite': True, 'elementwise_gate': True, 'relative_l2': .002, 'max_abs': .3}]},
        'loss_delta': {'candidate_graph': -.00005}}
    result = report.numerical_summary(quality, {'logit_relative_l2': .02, 'validation_loss_atol': .001})
    assert result['failed_blocks'] == result['elementwise_failure_blocks'] == 1
    assert result['nonfinite_blocks'] == result['relative_l2_failure_blocks'] == 0
    assert result['loss_gate'] is True
