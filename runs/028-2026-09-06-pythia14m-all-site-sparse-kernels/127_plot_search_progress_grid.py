"""Requested grid view; preserve Figure06 and its source/provenance unchanged."""
import importlib.util

from common import RUN, read_json, record, verify_record, write_json

spec = importlib.util.spec_from_file_location('run028_progress_original', RUN / '126_plot_search_progress.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def build_figure(data):
    fig, ax = original.build_figure(data)
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='both', color='#DEDEDE', linewidth=0.55)
    return fig, ax


def main():
    prior = read_json(RUN / 'results/search-progress-figure-001.json')
    for key in ['figure', 'source', 'script']:
        verify_record(prior[key])
    data = read_json(RUN / 'results/search-progress-001.json')
    verify_record(data['script'])
    fig, ax = build_figure(data)
    output = RUN / 'figures/07-autoresearch-kernel-progress-grid.pdf'
    fig.savefig(output, metadata={
        'Title': 'Auto-research progress: full-model graph speedup, K031-K050, grid view',
        'Creator': 'Run028/127_plot_search_progress_grid.py',
        'CreationDate': None, 'ModDate': None,
    })
    original.plt.close(fig)
    write_json(RUN / 'results/search-progress-grid-001.json', {
        **prior, 'figure': record(output), 'script': record(__file__),
        'base_script': prior['script'],
        'prior_provenance': record(RUN / 'results/search-progress-figure-001.json'),
        'presentation': 'Figure06 measurements/layout unchanged; light major-axis grid added',
    })
    print(record(output))


if __name__ == '__main__':
    main()
