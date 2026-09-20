"""Pairing, selection, fixed-checkpoint progress and single-plot regressions."""
import importlib.util
from pathlib import Path
import sys

import pytest

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, RUN / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reduce = load('run028_search_reduce', '125_reduce_search_progress.py')
plot = load('run028_search_plot', '126_plot_search_progress.py')


def test_raw_pairing_not_ratio_of_pooled_medians():
    samples = [{'mode': mode, 'input_index': i, 'repeat': 0, 'host_ms': value,
                'output_shape': [1, 2048, 50304]}
               for i, values in enumerate([(2, 1), (30, 10)])
               for mode, value in zip(['a', 'b'], values)]
    assert reduce.paired_speedup(samples, 'a', 'b', 2, 1) == pytest.approx(6 ** 0.5)
    for invalid in [samples[:-1], samples + [samples[0]]]:
        with pytest.raises(ValueError):
            reduce.paired_speedup(invalid, 'a', 'b', 2, 1)


def point(k=30, speed=2, **kw):
    return dict(iteration=k, speedup=speed, execution='graph', condition='c30',
                qualified=True, validation_blocks=338, stage='development',
                inputs=32, passes=7, processes=1, retry=1, attempt=f'k{k}', **kw)


def test_selection_uses_coverage_not_speed_or_qualification():
    early = point(speed=9)
    final = dict(early, stage='final', speedup=1.5, qualified=False, attempt='final', processes=3)
    selected, superseded = reduce.choose_points([early, final])
    assert selected == [final] and superseded == [early]


def test_best_line_keeps_checkpoint_and_execution_fixed():
    data = [point(k=31, speed=1.3), point(k=32, speed=1.2),
            dict(point(k=33, speed=9), qualified=False),
            dict(point(k=34, speed=8), condition='c01'),
            dict(point(k=35, speed=7), execution='eager'), point(k=36, speed=1.4)]
    trace = reduce.fixed_checkpoint_best(data, 'graph')
    assert [p['speedup'] for p in trace] == [1.3, 1.3, 1.3, 1.3, 1.4]


def test_single_metric_single_axes_all_model_points_no_bars_or_grid():
    data = reduce.read_json(RUN / 'results/search-progress-001.json')
    fig, ax = plot.build_figure(data)
    try:
        assert fig.axes == [ax] and not fig.legends and ax.get_legend() is None
        assert not ax.patches and not ax.containers
        assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
        assert [len(c.get_offsets()) for c in ax.collections] == [135, 1]
        plotted = [tuple(xy) for c in ax.collections for xy in c.get_offsets()]
        expected = [(p['iteration'], p['speedup']) for p in data['points'] if p['execution'] == 'graph']
        assert sorted(plotted) == sorted(expected)
        assert len(ax.lines) == 1
        assert list(ax.lines[0].get_ydata()) == [p['speedup'] for p in data['best_fixed_c30']['graph']]
    finally:
        plot.plt.close(fig)


def test_actual_history_preserves_all_iterations_and_mode_separation():
    data = reduce.read_json(RUN / 'results/search-progress-001.json')
    points = data['points']
    assert {p['iteration'] for p in points} == set(range(20, 51))
    assert len({(p['iteration'], p['execution'], p['condition']) for p in points}) == len(points)
    for k in [36, 50]:
        for mode in ['eager', 'graph']:
            group = [p for p in points if p['iteration'] == k and p['execution'] == mode]
            assert len(group) == 35 and all(p['stage'] == 'final' and p['qualified'] for p in group)
    assert all(p['iteration'] >= 31 for p in points if p['execution'] == 'graph')
    assert any(not p['qualified'] for p in points)
    assert data['best_fixed_c30']['graph'][-1]['speedup'] == pytest.approx(1.738989802462756)
