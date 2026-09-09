"""Compact displays of unchanged records for the nine-page main text."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('revised019', HERE / '05_make_figures.py')
revised = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revised)


def geometry(axes):
    """Record plotted numeric geometry independently of layout and annotation."""
    return [dict(
        limits=[list(ax.get_xlim()), list(ax.get_ylim())],
        lines=[line.get_xydata().tolist() for line in ax.lines],
        collections=[dict(offsets=c.get_offsets().tolist(),
                          paths=[p.vertices.tolist() for p in c.get_paths()])
                     for c in ax.collections],
        patches=[p.get_path().vertices.tolist() for p in ax.patches],
    ) for ax in axes]


CHECKS = {}


def save_compact(fig, name):
    selected = fig.axes[-2:] if name == '05-v4-activation-density-grid.pdf' else fig.axes[:]
    before = geometry(selected)
    if name == '01-v2-14m-overview.pdf':
        name = '01-v3-14m-overview-compact.pdf'
        fig.set_size_inches(5.5, 3.45)
        fig.subplots_adjust(left=.12, right=.98, bottom=.35, top=.87)
        fig.legends[0].set_bbox_to_anchor((.55, .235))
        fig.legends[1].set_bbox_to_anchor((.55, .005))
    elif name == '02-blocked-intervention-effects.pdf':
        name = '02-v3-paired-effects-compact.pdf'
        fig.set_size_inches(5.5, 4.0)
        fig.subplots_adjust(left=.16, right=.99, bottom=.34, top=.90, hspace=.13)
    elif name == '05-v4-activation-density-grid.pdf':
        name = '05-v5-activation-density-high-threshold.pdf'
        for ax in fig.axes[:-2]:
            fig.delaxes(ax)
        fig.set_size_inches(5.5, 3.1)
        for j, ax in enumerate(selected):
            ax.set_position([.12 + j*.48, .28, .38, .44])
            ax.set_title('FFN (h, m)' if j == 0 else 'Attention (q, k, v)',
                         fontsize=9, y=1.32, pad=0)
        fig._suptitle.set_text(r'Zero mass and nonzero distributions at $\kappa=0.5$')
    else:
        raise ValueError(name)
    assert geometry(selected) == before, 'Layout changed numeric geometry'
    CHECKS[name] = {'numeric_geometry_unchanged': True,
                    'axes': len(selected),
                    'geometry_sha256': hashlib.sha256(
                        json.dumps(before, sort_keys=True).encode()).hexdigest(),
                    'selection': 'kappa=0.5 only; full grid retained in appendix'
                                 if 'density' in name else 'all original plotted points'}
    fig.savefig(HERE/'figures'/name,
                metadata={'Creator': 'Analysis 019; condensed existing-record display',
                          'CreationDate': None, 'ModDate': None})
    revised.plt.close(fig)


def main():
    revised.plots.configure()
    revised.save = save_compact
    revised.plots.save = revised.revised_original
    revised.plots.overview(revised.DATA, all_variants=True)
    revised.plots.effects(revised.DATA)
    revised.density_figure()
    sources = [HERE/'05_make_figures.py', Path(revised.plots.__file__),
               revised.PRIOR/'figure_data.json',
               revised.PRIOR/'activation-density-v3-data.json',
               *sorted((ROOT/'manuscript/draft/supplementary-data/histograms').glob('*.json.gz'))]
    (HERE/'figures/CONDENSED-SOURCES.json').write_text(json.dumps({
        'sources': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources}, 'checks': CHECKS}, indent=2)+'\n',
        encoding='utf-8', newline='\n')
    print('Three compact PDFs; original plotted numeric geometry unchanged.')


if __name__ == '__main__':
    main()
