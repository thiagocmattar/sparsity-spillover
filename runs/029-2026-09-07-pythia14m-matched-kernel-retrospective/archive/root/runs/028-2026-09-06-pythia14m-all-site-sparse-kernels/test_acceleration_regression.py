"""Regression math, units, and complete-cohort checks for the Figure05 view."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('run028_acceleration_regression', RUN / '124_acceleration_regression.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_line_and_fraction_units():
    x = np.array([0, 5, 10, 20, 30])
    fit = module.fit_ols(x, 1 + 0.03 * x)
    assert fit['intercept'] == pytest.approx(1)
    assert fit['slope_per_percentage_point'] == pytest.approx(0.03)
    assert fit['slope_per_fraction'] == pytest.approx(3)
    assert fit['r_squared'] == pytest.approx(1)


@pytest.mark.parametrize('x,y', [([0, 1], [1, 2]), ([1, 1, 1], [1, 2, 3]),
                               ([0, 1, 2], [1, 1, 1]), ([0, 1, 2], [1, np.nan, 3])])
def test_invalid_fit_inputs(x, y):
    with pytest.raises(ValueError):
        module.fit_ols(x, y)


def test_actual_complete_cohort_and_independent_fit():
    data = module.read_json(RUN / 'results/summary-002.json')
    points = module.cohort_points(data)
    assert len(points) == 35
    x = np.array([p['R_model_percent'] for p in points])
    y = np.array([p['speedup'] for p in points])
    fit = module.fit_ols(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit['slope_per_percentage_point'] == pytest.approx(slope)
    assert fit['intercept'] == pytest.approx(intercept)
    assert fit['r_squared'] == pytest.approx(float(np.corrcoef(x, y)[0, 1] ** 2))
    assert sum(fit['residuals']) == pytest.approx(0, abs=1e-12)
    for change in ('missing', 'duplicate', 'unqualified', 'counts'):
        bad = copy.deepcopy(data)
        if change == 'missing':
            bad['rows'].pop()
        elif change == 'duplicate':
            bad['rows'][-1] = bad['rows'][0]
        elif change == 'unqualified':
            bad['rows'][0]['comparisons']['graph']['qualified'] = False
        else:
            bad['rows'][0]['R_model_percent'] += 1
        with pytest.raises(ValueError):
            module.cohort_points(bad)
