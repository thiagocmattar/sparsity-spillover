"""Paper-layout checks against the actual retained reduction; no synthetic research data."""
from pathlib import Path
import sys

import numpy as np
import pytest
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import PercentFormatter

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
from io_utils import module, read, verify
plot = module('run029_paper_summary_test', RUN / '18_paper_summary.py')


@pytest.fixture
def summary():
    data = read(verify(read(RUN / 'results/figures-002.json')['source']))
    catalog = read(RUN / 'provenance/candidates.json')['configurations']
    fig, axes = plot.paper_figure(data, catalog)
    yield fig, axes, data, catalog
    plot.plt.close(fig)


def test_paper_layout_labels_and_no_complex_legend(summary):
    fig, (left, right), _, _ = summary
    assert len(fig.axes) == 2
    assert left.get_subplotspec().get_gridspec().get_geometry() == (1, 2)
    assert left.get_xlabel() == 'Kernel iteration'
    assert right.get_xlabel() == r'$R_{\mathrm{model}}$'
    assert 'auto-research' in fig._suptitle.get_text()
    assert 'Kernel auto-research progress' in left.get_title()
    assert 'Best kernel (final iteration 42)' in right.get_title()
    assert 'increases with sparsity' in right.get_title()
    assert not left.texts
    for ax in (left, right):
        assert ax.get_legend() is None and not ax.containers
        assert ax.get_yscale() == 'linear'
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert any(line.get_visible() for line in ax.get_ygridlines())


def test_all_timed_history_points_are_circles_without_requalification(summary):
    _, (left, _), data, catalog = summary
    rows = [p for p in data['points'] if p['phase'] == 'history' and 'speedup' in p]
    mapping = {c['id']: c.get('paper_iteration', 0) for c in catalog if c['status'] == 'eligible'}
    assert len(rows) == 214
    assert sum(p['qualified'] for p in rows) == 140
    assert len(left.collections) == 1
    dots = left.collections[0]
    np.testing.assert_array_equal(dots.get_offsets(), [[mapping[p['candidate']], p['speedup']] for p in rows])
    marker = MarkerStyle('o')
    np.testing.assert_array_equal(dots.get_paths()[0].vertices,
                                  marker.get_path().transformed(marker.get_transform()).vertices)
    assert len(dots.get_sizes()) == len(dots.get_facecolors()) == 1
    np.testing.assert_array_equal(left.lines[0].get_xdata(), [p['iteration'] for p in data['progress']])
    np.testing.assert_array_equal(left.lines[0].get_ydata(), [p['speedup'] for p in data['progress']])
    assert data['progress'][0]['speedup'] == 1
    assert data['progress'][-1]['incumbent'] == 'k050'


def test_percentage_ticks_do_not_rescale_the_fit_or_points(summary):
    _, (_, right), data, _ = summary
    rows = [p for p in data['points'] if p['phase'] == 'final' and p['candidate'] == 'k050']
    assert len(rows) == 35 and all(p['qualified'] for p in rows)
    assert len(right.collections) == 1
    np.testing.assert_array_equal(right.collections[0].get_offsets(), [[p['R_model'], p['speedup']] for p in rows])
    formatter = right.xaxis.get_major_formatter()
    assert isinstance(formatter, PercentFormatter) and formatter(.1, 0) == '10%'
    line = right.lines[0]
    assert line.get_color() == 'black' and line.get_linestyle() == '--'
    fit = data['k050_regression']
    np.testing.assert_allclose(line.get_ydata(), fit['intercept'] + fit['slope'] * line.get_xdata())
    coefficients = np.polyfit([p['R_model'] for p in rows], [p['speedup'] for p in rows], 1)
    np.testing.assert_allclose(coefficients, [fit['slope'], fit['intercept']], rtol=1e-12)
    assert .9 < right.get_ylim()[0] < 1
    assert right.get_ylim()[1] > max(line.get_ydata())
    assert '0.781' in right.texts[0].get_text()
