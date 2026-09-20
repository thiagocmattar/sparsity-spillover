"""Check coverage, actual serialized values and measurement identity in the appendix."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class CompactResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((HERE / 'data/compact-results-appendix.json').read_text())
        cls.quality = json.loads((HERE / 'data/all-model-quality-sparsity.json').read_text())

    def test_disjoint_complete_partition_and_matched_settings(self):
        tables = self.data['tables']
        rows = [r for part in tables.values() for r in part]
        self.assertEqual(len(rows), 74)
        self.assertEqual(len({r['source_attempt'] for r in rows}), 74)
        self.assertEqual({r['source_attempt']: r for r in rows},
                         {r['source_attempt']: r for r in self.quality['trained_points']})
        shared = tables['14m-70m']
        for size in ['14M', '70M']:
            self.assertEqual(sum(r['model'] == size for r in shared), 22)
        keys = lambda size: {(r['scope'], r['pressure'], r['kappa']) for r in shared if r['model'] == size}
        self.assertEqual(keys('14M'), keys('70M'))
        self.assertEqual({r['model'] for r in tables['14m-only']}, {'14M'})
        self.assertEqual({r['model'] for r in tables['410m']}, {'410M'})

    def test_serialized_table_values_match_final_loss_and_pooled_counts(self):
        for name, records in self.data['tables'].items():
            source = (HERE / f'tables/compact-results/{name}.tex').read_text()
            actual = []
            for line in source.splitlines():
                if not line.endswith(r' \\'):
                    continue
                cells = line[:-3].split(' & ')
                if len(cells) < 4:
                    continue
                try:
                    if name in ['14m-only', '14m-70m']:
                        actual.extend([(float(cells[-4]), float(cells[-3])), (float(cells[-2]), float(cells[-1]))])
                    else:
                        actual.append((float(cells[-2]), float(cells[-1])))
                except ValueError:
                    continue
            expected = []
            for r in records:
                metrics = json.loads((ROOT / r['source_attempt'] / 'metrics.json').read_text())
                logical = json.loads((ROOT / r['source_attempt'] / 'diagnostics/logical_products.json').read_text())
                counts = logical['measured']
                expected.append((float(f"{metrics['validation']['final']['loss']:.4f}"),
                                 float(f"{100 * counts['block_zero_product_count'] / counts['model_product_count']:.3f}")))
            self.assertEqual(Counter(actual), Counter(expected), name)
            self.assertIn(r'\multirow', source)
            self.assertNotRegex(source, r'A[0147]|P_[47]|U_\{')

    def test_posthoc_points_keep_identity_full_coverage_and_actual_p0(self):
        raw = json.loads((ROOT / 'runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json').read_text())
        original = {p['id']: p for p in raw['points'] if p['scale'] == '14M'}
        points = self.data['posthoc_points']
        self.assertEqual(len(points), 300)
        self.assertEqual({p['id'] for p in points}, set(original))
        for p in points:
            source = original[p['id']]
            self.assertEqual(p['loss'], source['loss'])
            self.assertEqual(p['target'], source['dose'])
            self.assertEqual(p['checkpoint_sha256'], source['checkpoint_content_sha256'])
            self.assertEqual(p['zero_product_count'], source['counts']['block_zero_product_count'])
            self.assertEqual(p['model_product_count'], source['counts']['model_product_count'])
        for r in self.data['posthoc_trained']:
            path = [p for p in points if p['source_attempt'] == r['source_attempt']]
            self.assertEqual(sorted(p['target'] for p in path), [i / 10 for i in range(10)])
            self.assertEqual({p['checkpoint_sha256'] for p in path}, {r['final_checkpoint_content_sha256']})
        self.assertEqual(len(self.data['posthoc_unmeasured']), 10)
        self.assertEqual(self.data['figure']['panels'][1]['outside'], [])

    def test_artifact_hashes_and_manuscript_copies(self):
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path, digest in self.data['sources_sha256'].items():
            self.assertEqual(sha(ROOT / path), digest)
        for path, digest in self.data['table_sha256'].items():
            self.assertEqual(sha(HERE / path), digest)
            self.assertEqual(sha(ROOT / 'manuscript/draft' / path), digest)
        figure = self.data['figure']
        self.assertEqual(sha(HERE / figure['path']), figure['sha256'])
        self.assertEqual(sha(ROOT / 'manuscript/draft/figures/appendix' / Path(figure['path']).name), figure['sha256'])
        self.assertEqual(sha(HERE / '33_compact_results_appendix.py'), self.data['script_sha256'])


if __name__ == '__main__':
    unittest.main()
