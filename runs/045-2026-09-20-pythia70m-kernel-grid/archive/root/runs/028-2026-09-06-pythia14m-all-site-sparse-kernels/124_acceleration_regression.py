"""Single-panel OLS view of all 35 frozen K050 full-model graph speedups."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from common import RUN, read_json, record, verify_record, write_json


def fit_ols(x_percent, speedup):
    """Unweighted least squares with a freely estimated intercept, one row/variant."""
    x = np.asarray(x_percent, dtype=np.float64)
    y = np.asarray(speedup, dtype=np.float64)
    if (x.ndim != 1 or y.shape != x.shape or x.size < 3
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.ptp(x) == 0 or np.ptp(y) == 0):
        raise ValueError('OLS requires paired, finite, nonconstant vectors')
    design = np.column_stack((np.ones(x.size), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    residuals = y - predicted
    sse = float(residuals @ residuals)
    centered = y - y.mean()
    sst = float(centered @ centered)
    return {
        'method': 'Unweighted OLS with intercept; one paired-geomean point per variant',
        'n': int(x.size), 'intercept': float(intercept),
        'slope_per_percentage_point': float(slope),
        'slope_per_fraction': float(100 * slope),
        'r_squared': 1 - sse / sst, 'sse': sse, 'sst': sst,
        'rmse': float(np.sqrt(sse / x.size)),
        'predicted': predicted.tolist(), 'residuals': residuals.tolist(),
        'x_range_percent': [float(x.min()), float(x.max())],
    }


def cohort_points(data):
    rows = data['rows']
    if len(rows) != 35 or {r['condition'] for r in rows} != {f'c{i:02d}' for i in range(1, 36)}:
        raise ValueError('All 35 unique variants must be retained')
    points = []
    for row in rows:
        if not row['complete']:
            raise ValueError('Incomplete variant')
        comparison = row['comparisons']['graph']
        if not comparison['qualified']:
            raise ValueError('Unqualified graph comparison')
        x = 100 * row['canonical_zero_products'] / row['canonical_model_products']
        if x != row['R_model_percent']:
            raise ValueError('Canonical count/percentage mismatch')
        ratios = comparison['process_ratios']
        y = comparison['ratio']
        if (len(ratios) != 3 or not all(math.isfinite(v) and v > 0 for v in ratios)
                or not math.isclose(y, math.exp(np.log(ratios).mean()), rel_tol=1e-12)
                or comparison['process_min'] != min(ratios)
                or comparison['process_max'] != max(ratios)):
            raise ValueError('Invalid paired-process reduction')
        points.append({'condition': row['condition'], 'R_model_percent': x,
                       'speedup': y, 'process_min': min(ratios), 'process_max': max(ratios)})
    return points


def main():
    prior = read_json(RUN / 'results/figure-provenance-002.json')
    verify_record(prior['source'])
    data = read_json(RUN / 'results/summary-002.json')
    verify_record(data['policy'])
    points = cohort_points(data)
    x = np.array([p['R_model_percent'] for p in points])
    y = np.array([p['speedup'] for p in points])
    fit = fit_ols(x, y)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'axes.titlesize': 10.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'savefig.transparent': False,
    })
    fig, ax = plt.subplots(figsize=(5.7, 4.1))
    ax.axhline(1, color='#777777', linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.errorbar(x, y, yerr=[y - [p['process_min'] for p in points],
                            [p['process_max'] for p in points] - y],
                fmt='o', color='#0072B2', ecolor='#555555', markersize=4.8,
                markeredgecolor='white', markeredgewidth=0.35, capsize=1.8,
                elinewidth=0.7, alpha=0.9, zorder=3)
    line_x = np.linspace(x.min(), x.max(), 200)
    line_y = fit['intercept'] + fit['slope_per_percentage_point'] * line_x
    ax.plot(line_x, line_y, color='#D55E00', linewidth=1.5, zorder=2)
    ax.text(0.035, 0.96,
            rf"$\widehat{{S}} = {fit['intercept']:.3f} + {fit['slope_per_fraction']:.3f}\,R_{{\mathrm{{model}}}}$"
            '\n' + rf"$R^2 = {fit['r_squared']:.3f}\qquad n = {fit['n']}$",
            transform=ax.transAxes, va='top', fontsize=10,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.93, 'pad': 3})
    low = min(min(p['process_min'] for p in points), float(line_y.min()))
    high = max(max(p['process_max'] for p in points), float(line_y.max()))
    ax.set(xlim=(-0.7, 28.5), ylim=(math.floor((low - 0.01) * 20) / 20,
                                   math.ceil((high + 0.02) * 20) / 20),
           xlabel=r'Canonical $R_{\mathrm{model}}$ (%)',
           ylabel=r'Full-model speedup, $S$ ($\times$)')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis='y', color='#dddddd', linewidth=0.5)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.965, 'Full-model acceleration with a specialized kernel',
             ha='center', va='top', fontsize=10.5)
    fig.text(0.5, 0.904, 'Pythia-14M | K050 | RTX 5090 | BF16 | B = 1, T = 2,048',
             ha='center', fontsize=7.7)
    fig.text(0.5, 0.077, 'Native graph / K050 graph; 3 processes x 64 inputs x 7 paired passes.',
             ha='center', fontsize=7)
    fig.text(0.5, 0.044, 'Bars: process min-max. OLS uses fractional R in equation. Expanded y-axis.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.22, top=0.865)
    output = RUN / 'figures/05-hybrid-acceleration-linear-fit.pdf'
    fig.savefig(output, metadata={'Title': 'Canonical R_model and K050 full-model speedup: linear fit',
                                 'Creator': 'Run028/124_acceleration_regression.py',
                                 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    write_json(RUN / 'results/acceleration-regression-001.json', {
        'fit': fit, 'points': points, 'coverage': data['coverage'],
        'source': record(RUN / 'results/summary-002.json'), 'policy': data['policy'],
        'script': record(__file__), 'figure': record(output),
        'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
        'limits': 'Descriptive in-sample association, not held-out prediction or causal attribution. '
                  'Canonical FP16 logical R; BF16 timing. Same data as Figure03 panel(a), no exclusions. '
                  'No confidence band; process bars are not regression uncertainty. '
                  'Topologies, weights, sparsity patterns, gate costs and quality differ across variants.',
    })
    print({k: v for k, v in fit.items() if k not in {'predicted', 'residuals'}})


if __name__ == '__main__':
    main()
