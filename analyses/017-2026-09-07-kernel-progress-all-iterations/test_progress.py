import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reduce = load('analysis017_test_reduce', '01_reduce.py')
plot = load('analysis017_test_plot', '02_plot.py')


def test_paired_geomean_and_reject_missing_duplicate_wrong_workload():
    samples = [{'mode': mode, 'input_index': i, 'repeat': 0, 'host_ms': value,
                'output_shape': [1, 2048, 50304]}
               for i, values in enumerate([(2, 1), (30, 10)])
               for mode, value in zip(['a', 'b'], values)]
    assert reduce.paired(samples, 'a', 'b', 2, 1) == pytest.approx(6 ** .5)
    for broken in [samples[:-1], samples + [samples[0]],
                   [dict(samples[0], output_shape=[1, 1, 50304])] + samples[1:]]:
        with pytest.raises(ValueError):
            reduce.paired(broken, 'a', 'b', 2, 1)


def test_retries_selected_by_coverage_not_speed_or_correctness():
    point = dict(iteration=11, condition='a', setting='suffix3', checked_blocks=16,
                 inputs=16, passes=10, attempt='k011search', speedup=9, qualified=True)
    final = dict(point, checked_blocks=338, passes=3, attempt='k011full', speedup=.9, qualified=False)
    other_mask = dict(point, setting='suffix4')
    selected, superseded = reduce.choose_early([point, final, other_mask])
    assert final in selected and other_mask in selected and point in superseded


def test_incumbent_does_not_cross_protocols_or_admit_failed_partial_checks():
    base = dict(iteration=1, speedup=9, phase='a', anchor=True, qualified=True, validation_blocks=338)
    points = [base, dict(base, iteration=2, phase='b', speedup=1.2),
              dict(base, iteration=3, phase='b', qualified=False),
              dict(base, iteration=4, phase='b', validation_blocks=16),
              dict(base, iteration=5, phase='b', anchor=False)]
    assert [p['speedup'] for p in reduce.incumbent(points, 'b')] == [1.2] * 4


def test_actual_history_missing_is_not_zero_and_graph_points_unchanged():
    data = reduce.read(HERE / 'results/progress.json')
    assert len(data['points']) == 253
    assert sum(p['qualified'] for p in data['points']) == 193
    assert data['missing_iterations'] == [2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16]
    assert all(p['speedup'] > 0 and 1 <= p['iteration'] <= 50 for p in data['points'])
    for p in data['points']:
        assert p['phase'] == ('pro4500_eager' if p['iteration'] < 17 else '5090_eager' if p['iteration'] < 31 else '5090_graph')
    late = reduce.read(reduce.R28 / 'results/search-progress-001.json')
    expected = [(p['iteration'], p['condition'], p['speedup'], p['qualified']) for p in late['points'] if p['execution'] == 'graph']
    actual = [(p['iteration'], p['condition'], p['speedup'], p['qualified']) for p in data['points'] if p['phase'] == '5090_graph']
    assert sorted(actual) == sorted(expected)


def test_single_plot_retains_every_point_with_grid_no_bars_or_legend():
    data = reduce.read(HERE / 'results/progress.json')
    fig, ax = plot.build_figure(data)
    try:
        assert len(fig.axes) == 1 and not ax.containers and not ax.patches
        assert not fig.legends and ax.get_legend() is None
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert any(line.get_visible() for line in ax.get_ygridlines())
        assert sorted(tuple(xy) for c in ax.collections for xy in c.get_offsets()) == sorted((p['iteration'], p['speedup']) for p in data['points'])
        assert len(ax.lines) == 5  # Three phase-local step traces, two protocol boundaries.
        assert ax.get_xlim()[0] < 1 and ax.get_xlim()[1] > 50
    finally:
        plot.plt.close(fig)
