"""Evidence checks for the two-size Figure 6 extension and measured clipping join."""
import hashlib
import json
import math
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


class MatchedQualityLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.figure=json.loads((HERE/'data/14m-70m-quality-sparsity-latency.json').read_text())
        cls.table=json.loads((HERE/'data/paper-checkpoints.json').read_text())
        cls.timing=json.loads((ROOT/'runs/036-2026-09-18-controls-clipping-final-kernel/results/clipping-final-kernel.json').read_text())

    def test_source_and_output_identity(self):
        for path,digest in self.figure['sources_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest)
        for key in ['script','output']:
            self.assertEqual(hashlib.sha256((HERE/self.figure[key]).read_bytes()).hexdigest(),self.figure[key+'_sha256'])

    def test_matched_trained_cohorts_preserve_figure6(self):
        records=self.figure['trained_points']
        self.assertEqual(len(records),44)
        source={r['checkpoint_key']:r for r in self.table['checkpoints']}
        for row in records:
            self.assertTrue(source[row['checkpoint_key']]['comparison_cohort'])
            self.assertEqual(row,{k:source[row['checkpoint_key']][k] for k in row})
        for size in ['14M','70M']:
            rows=[r for r in records if r['model']==size]
            self.assertEqual(len(rows),22)
            for scope in ['4','7']:
                for pressure in ['h','all']:
                    self.assertEqual(sorted(r['kappa'] for r in rows if (r['scope'],r['pressure'])==(scope,pressure)),[0,.01,.05,.1,.5])
            self.assertEqual({r['scope'] for r in rows if r['pressure']=='none'},{'0','1'})
            panels=[p for p in self.figure['panels'] if p['model']==size]
            self.assertEqual(panels[0]['trained_keys'],panels[1]['trained_keys'])
        previous=json.loads((HERE/'data/14m-quality-sparsity-latency.json').read_text())
        old={r['checkpoint_key']:r for r in previous['trained_points']}
        now={r['checkpoint_key']:r for r in records if r['model']=='14M'}
        self.assertEqual(set(old),set(now))
        for key,row in old.items():
            self.assertEqual(row,{k:now[key][k] for k in row})

    def test_clipping_identity_counts_and_geometric_means(self):
        points=self.figure['clipping_points']
        self.assertEqual(len(points),40)
        quality={r['id']:r for r in self.table['clipping'] if r['main_clipping']}
        runtime={r['condition']:r for r in self.timing['points']}
        self.assertEqual({p['id'] for p in points},set(quality))
        self.assertEqual({p['condition'] for p in points},set(runtime))
        for point in points:
            q,t=quality[point['id']],runtime[point['condition']]
            self.assertEqual((point['checkpoint_key'],point['target']),(q['checkpoint_key'],q['target']))
            self.assertEqual((point['checkpoint_key'],point['target']),(t['checkpoint_key'],t['p']))
            self.assertEqual(point['loss'],q['loss'])
            self.assertEqual(point['loss'],t['retained_fp16_loss'])
            self.assertTrue(t['qualified'])
            self.assertEqual(len(t['replicates']),3)
            self.assertAlmostEqual(point['latency_ms'],math.exp(sum(math.log(r['candidate_gm_ms']) for r in t['replicates'])/3),places=12)
            counts=t['retained_fp16_counts']
            self.assertEqual(counts,q['counts'])
            self.assertEqual(point['zero_product_count'],counts['block_zero_product_count'])
            self.assertEqual(point['model_product_count'],counts['model_product_count'])
            self.assertAlmostEqual(point['sparsity'],100*point['zero_product_count']/point['model_product_count'],places=12)
            self.assertEqual(point['coverage']['sequences'],338)
            self.assertEqual(point['coverage']['excluded_tail_tokens'],1444)

    def test_displayed_latency_and_disclosed_quality_tails(self):
        points={p['id']:p for p in self.figure['clipping_points']}
        for panel in self.figure['panels']:
            lo,hi=self.figure['limits'][panel['model']][panel['metric']]
            outside=[key for key in panel['clipping_ids'] if not lo<=points[key][panel['metric']]<=hi]
            self.assertEqual(panel['clipping_outside_y'],outside)
            self.assertEqual(len(panel['clipping_ids']),20)
            self.assertEqual(len(outside),0 if panel['metric']=='latency_ms' else {'14M':8,'70M':7}[panel['model']])
            xlo,xhi=self.figure['limits'][panel['model']]['sparsity']
            self.assertTrue(all(xlo<=points[key]['sparsity']<=xhi for key in panel['clipping_ids']))


if __name__=='__main__':
    unittest.main()
