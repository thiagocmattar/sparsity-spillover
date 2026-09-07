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
