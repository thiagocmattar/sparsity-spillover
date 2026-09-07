"""The grid view must not change the original figure or measured points."""
import importlib.util
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('run028_progress_grid', RUN / '127_plot_search_progress_grid.py')
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def test_grid_view_preserves_points_and_incumbent():
    data = grid.read_json(RUN / 'results/search-progress-001.json')
    fig, ax = grid.build_figure(data)
    try:
        assert len(fig.axes) == 1 and not ax.patches and not ax.containers
        assert not fig.legends and ax.get_legend() is None
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert any(line.get_visible() for line in ax.get_ygridlines())
        actual = sorted(tuple(xy) for c in ax.collections for xy in c.get_offsets())
        expected = sorted((p['iteration'], p['speedup']) for p in data['points'] if p['execution'] == 'graph')
        assert actual == expected and len(actual) == 136
        assert list(ax.lines[0].get_ydata()) == [p['speedup'] for p in data['best_fixed_c30']['graph']]
    finally:
        grid.original.plt.close(fig)
