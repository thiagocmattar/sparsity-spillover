from pathlib import Path
import sys
import pytest

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN))
from io_utils import module
plot=module('run029_figure_test',RUN/'12_figures.py')


@pytest.fixture
def data():
    # Synthetic unit-test fixture only; never serialized as a research result.
    return {'progress':[{'iteration':i,'speedup':1 if i<5 else 1.4,'incumbent':'test'} for i in range(43)],
            'points':[{'phase':'final','candidate':'k050','condition':str(i),'R_model':i*.1,'speedup':1+i*.2,'qualified':True} for i in range(4)],
            'k050_regression':{'intercept':1.,'slope':2.,'r_squared':1.}}


def check_axes(fig,ax):
    assert len(fig.axes)==1 and not ax.containers
    assert ax.get_legend() is None
    assert any(line.get_visible() for line in ax.get_xgridlines())
    assert any(line.get_visible() for line in ax.get_ygridlines())
    assert ax.get_yscale()=='linear'


def test_progress_preserves_incumbent_and_iteration(data):
    fig,ax=plot.progress_figure(data);check_axes(fig,ax)
    assert list(ax.lines[0].get_xdata())==list(range(43))
    assert list(ax.lines[0].get_ydata())==[r['speedup'] for r in data['progress']]
    assert ax.get_ylim()[0]>.95
    plot.plt.close(fig)


def test_rmodel_uses_native_scale_and_uniform_markers(data):
    fig,ax=plot.rmodel_figure(data);check_axes(fig,ax)
    assert ax.get_ylim()[0]>.8
    assert ax.collections[0].get_offsets().tolist()==[[i*.1,1+i*.2] for i in range(4)]
    assert len(ax.collections[0].get_sizes())==1
    plot.plt.close(fig)


def test_individual_points_keep_failures_repeated_iterations_and_p0(data):
    catalog = [{'id': 'p0', 'status': 'eligible'},
               {'id': 'k011-prefix1', 'status': 'eligible', 'paper_iteration': 3},
               {'id': 'k011-prefix2', 'status': 'eligible', 'paper_iteration': 3},
               {'id': 'unsupported', 'status': 'eligible', 'paper_iteration': 4}]
    data['points'] = [
        {'phase': 'history', 'candidate': c, 'qualified': ok, 'speedup': s}
        for c, s, ok in [('p0', .8, True), ('k011-prefix1', 1.1, True),
                          ('k011-prefix2', 2., False)]]
    data['points'].append({'phase': 'history', 'candidate': 'unsupported', 'qualified': False})
    fig, ax = plot.individual_figure(data, catalog)
    check_axes(fig, ax)
    assert ax.collections[0].get_offsets().tolist() == [[0, .8], [3, 1.1]]
    assert ax.collections[1].get_offsets().tolist() == [[3, 2.]]
    assert ax.get_ylim()[0] < .8 and ax.get_ylim()[1] > 2.
    assert ax.yaxis.get_major_formatter()(.25, 0) == '0.25'
    plot.plt.close(fig)


def test_regression_endpoint_is_visible(data):
    data['k050_regression']['slope'] = 4.
    fig, ax = plot.rmodel_figure(data)
    assert ax.get_ylim()[1] > 2.2
    plot.plt.close(fig)
