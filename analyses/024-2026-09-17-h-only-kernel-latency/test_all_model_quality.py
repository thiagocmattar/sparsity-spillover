"""Scientific coverage and normalization checks for the three-size overview."""
import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class AllModelQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((HERE / 'data/all-model-quality-sparsity.json').read_text())

    def test_all_executed_endpoints_and_uniform_final_loss(self):
        rows = self.data['trained_points']
        self.assertEqual(len({p['source_attempt'] for p in rows}), 74)
        self.assertEqual([sum(p['model'] == s for p in rows) for s in ['14M', '70M', '410M']], [40, 22, 12])
        for p in rows:
            raw = json.loads((ROOT / p['source_attempt'] / 'metrics.json').read_text())
            logical = json.loads((ROOT / p['source_attempt'] / 'diagnostics/logical_products.json').read_text())
            self.assertEqual(p['loss'], raw['validation']['final']['loss'])
            counts = logical['measured']
            self.assertEqual(p['sparsity'], 100 * counts['block_zero_product_count'] / counts['model_product_count'])
            self.assertEqual(raw['training']['optimizer_step_count'], 712)
            self.assertEqual(raw['training']['input_tokens'], 1493172224)
            self.assertEqual(raw['validation']['final']['sequences'], 338)
            self.assertEqual(raw['validation']['final']['excluded_tail_tokens'], 1444)
        paper = json.loads((HERE / 'data/paper-checkpoints.json').read_text())
        by_attempt = {p['source_attempt']: p for p in rows}
        for p in paper['checkpoints']:
            self.assertEqual(by_attempt[p['source_attempt']]['loss'], p['loss'])
            self.assertEqual(by_attempt[p['source_attempt']]['sparsity'], p['sparsity'])

    def test_recipe_coverage_and_distinct_pressure_methods(self):
        series = self.data['series']
        self.assertEqual([sum(p['model'] == s for p in series) for s in ['14M', '70M', '410M']], [10, 6, 4])
        small = [p for p in series if p['model'] == '14M']
        self.assertEqual(len({p['color'] for p in small}), 10)
        self.assertEqual({p['pressure'] for p in small if p['scope'] == '1'}, {'none', 'L1', 'h'})
        drawn = [a for s in series for a in s['source_attempts']]
        self.assertEqual(len(drawn), 74)
        self.assertEqual(set(drawn), {p['source_attempt'] for p in self.data['trained_points']})

    def test_clipping_coverage_visibility_and_ceilings(self):
        rows, clips = self.data['trained_points'], self.data['clipping_points']
        self.assertEqual(len(clips), 60)
        lo, hi = self.data['shared_y_limits']
        self.assertTrue(all(lo <= p['loss'] <= hi for p in rows))
        self.assertEqual([len(p['clipping_outside_y']) for p in self.data['panels']], [8, 5, 7])
        for size in ['14M', '70M', '410M']:
            for scope in ['0', '1']:
                group = [p for p in clips if p['model'] == size and p['scope'] == scope]
                self.assertEqual(sorted(p['target'] for p in group), [p / 10 for p in range(10)])
                baseline = next(p for p in rows if p['model'] == size and p['scope'] == scope and p['pressure'] == 'none')
                self.assertTrue(all(p['checkpoint_content_sha256'] == baseline['final_checkpoint_content_sha256'] for p in group))
            for c in self.data['ceilings'][size].values():
                self.assertAlmostEqual(c['R_model_max_percent'], 100 * c['reachable_product_count'] / c['model_product_count'])

    def test_sources_and_speedup_denominator(self):
        for path, digest in self.data['sources_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        self.assertEqual(hashlib.sha256((HERE / self.data['output']).read_bytes()).hexdigest(), self.data['output_sha256'])
        self.assertEqual(hashlib.sha256((HERE / self.data['script']).read_bytes()).hexdigest(), self.data['script_sha256'])
        rows = self.data['trained_points']
        for point in self.data['high_threshold_speedups']:
            baseline = next(r for r in rows if r['model'] == point['model'] and r['scope'] == '0')
            self.assertEqual(point['base_latency_ms'], baseline['latency_ms'])
            self.assertEqual(point['speedup_vs_optimized_base'], baseline['latency_ms'] / point['latency_ms'])
        for scope in ['4', '7']:
            for pressure in ['h', 'all']:
                pair = {p['model']: p for p in self.data['high_threshold_speedups'] if p['scope'] == scope and p['pressure'] == pressure}
                self.assertGreater(pair['70M']['sparsity'], pair['14M']['sparsity'])
                self.assertGreater(pair['70M']['speedup_vs_optimized_base'], pair['14M']['speedup_vs_optimized_base'])


if __name__ == '__main__':
    unittest.main()
