"""Two-panel paper figure from the frozen matched retrospective, without new timing."""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from io_utils import RUN, read, record, verify, write

BLUE = '#176b91'
GRAY = '#6d7378'
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9,
    'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65,
    'savefig.dpi': 300,
}


def paper_figure(data, catalog):
    """Keep measured coordinates and qualification unchanged; simplify display only."""
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 3.25), layout='constrained')
        fig.get_layout_engine().set(w_pad=.035, h_pad=.045, wspace=.065)
        fig.suptitle('Sparse kernel auto-research for full-model acceleration', fontsize=11)
        for ax in (left, right):
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, which='major', color='#dadde0', linewidth=.5)
            ax.set_axisbelow(True)
            ax.tick_params(direction='out', length=3, width=.65)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.set_ylabel('Full-model speedup (\u00d7)')

        mapping = {row['id']: row.get('paper_iteration', 0) for row in catalog
                   if row['status'] == 'eligible'}
        history = [point for point in data['points']
                   if point['phase'] == 'history' and 'speedup' in point]
        left.scatter([mapping[p['candidate']] for p in history],
                     [p['speedup'] for p in history], marker='o',
                     s=13, color=GRAY, alpha=.55, linewidths=0, zorder=2)
        curve = data['progress']
        last = curve[-1]['iteration']
        left.step([p['iteration'] for p in curve], [p['speedup'] for p in curve],
                  where='post', color=BLUE, lw=1.6, zorder=3)
        left.axhline(1, color=GRAY, lw=.7, ls=(0, (3, 3)), zorder=1)
        ys = [p['speedup'] for p in history] + [p['speedup'] for p in curve]
        span = max(max(ys) - min(ys), .2)
        left.set_ylim(max(0, min(ys) - .04 * span), max(ys) + .08 * span)
        left.set_xlim(-1, last + 1)
        left.set_xticks([0, 10, 20, 30, last], ['P0', '10', '20', '30', str(last)])
        left.set_xlabel('Kernel iteration')
        left.set_title('(a) Kernel auto-research progress\n', pad=9, linespacing=1.4)

        final = [p for p in data['points'] if p['phase'] == 'final'
                 and p['candidate'] == 'k050' and 'speedup' in p]
        if not final or not all(p['qualified'] for p in final):
            raise ValueError('This paper view requires the fully qualified K050 cohort')
        right.scatter([p['R_model'] for p in final], [p['speedup'] for p in final],
                      marker='o', s=21, color=BLUE, alpha=.85, linewidths=0, zorder=3)
        fit = data['k050_regression']
        endpoints = [min(p['R_model'] for p in final), max(p['R_model'] for p in final)]
        fitted = [fit['intercept'] + fit['slope'] * x for x in endpoints]
        right.plot(endpoints, fitted, color='black', ls='--', lw=1.1, zorder=2)
        right.text(.045, .96,
                   rf'$\widehat{{S}} = {fit["intercept"]:.3f} + {fit["slope"]:.3f}\,R_{{\mathrm{{model}}}}$'
                   + '\n' + rf'$R^2 = {fit["r_squared"]:.3f}$',
                   transform=right.transAxes, va='top', fontsize=9, linespacing=1.5)
        ys = [p['speedup'] for p in final]
        span = max(max(ys) - min(ys), .2)
        right.set_ylim(min(ys) - .09 * span, max(ys + fitted) + .13 * span)
        right.xaxis.set_major_locator(MultipleLocator(.05))
        right.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        right.set_xlabel(r'$R_{\mathrm{model}}$')
        right.set_title(f'(b) Best kernel (final iteration {last})\n'
                        'Speedup increases with sparsity', pad=9, linespacing=1.4)
        return fig, (left, right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    args = parser.parse_args()
    if args.revision < 1:
        raise ValueError('Positive publication revision required')
    suffix = '' if args.revision == 1 else f'-r{args.revision:02d}'
    dest = RUN / f'figures/04-kernel-autoresearch-and-rmodel{suffix}.pdf'
    manifest = RUN / f'results/figures-summary-{args.revision:03d}.json'
    if dest.exists() or manifest.exists():
        raise ValueError('Existing publication record; use a new revision')
    previous = read(RUN / 'results/figures-002.json')
    path = verify(previous['source'])
    data = read(path)
    catalog_path = RUN / 'provenance/candidates.json'
    catalog = read(catalog_path)['configurations']
    fig, _ = paper_figure(data, catalog)
    # The PDF backend reads fonttype at save time, after paper_figure's context exits.
    with plt.rc_context(STYLE):
        fig.savefig(dest, metadata={
            'Creator': 'Run029 / Matplotlib',
            'Title': 'Sparse kernel auto-research for full-model acceleration',
            'Subject': 'Matched Pythia14M / RTX5090 retrospective; see Observation05 for caption and qualification.',
        })
    plt.close(fig)
    history = [p for p in data['points'] if p['phase'] == 'history']
    write(manifest, {
        'source': record(path), 'catalog': record(catalog_path), 'script': record(__file__),
        'figure': record(dest), 'k050_regression': data['k050_regression'],
        'history_timed_points': sum('speedup' in p for p in history),
        'history_qualified_points': sum(p['qualified'] for p in history),
        'history_unqualified_timed_points': sum('speedup' in p and not p['qualified'] for p in history),
        'history_untimed_points': sum('speedup' not in p for p in history),
        'final_qualified_points': sum(p['phase'] == 'final' and p['candidate'] == 'k050'
                                      and p['qualified'] for p in data['points']),
        'progress': data['progress'],
        'display': '1x2 vector PDF; regular history circles include numerical failures; '
                   'incumbent remains qualified-only on fixed c30; no history annotations or legend; '
                   'R_model percentage ticks, fractional regression units; black dashed OLS; linear speedups.',
    })
    print(record(dest))


if __name__ == '__main__':
    main()
