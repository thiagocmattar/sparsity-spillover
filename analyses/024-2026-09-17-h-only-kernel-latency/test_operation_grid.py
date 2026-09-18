"""Check the six-panel absolute operation decomposition against retained counts."""
import hashlib
import json
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent


class OperationGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.figure=json.loads((HERE/'data/14m-70m-operation-contributions.json').read_text())
        cls.source=json.loads((HERE/'data/paper-checkpoints.json').read_text())
        cls.reference=next(f for f in json.loads((HERE/'data/paper-derived.json').read_text())['figures']
                           if f['file']=='04-operation-sparsity-changes.pdf')

    def test_hashes_palette_and_preserved_reference_points(self):
        for name,digest in self.figure['sources_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
        for key in ['script','output']:
            self.assertEqual(hashlib.sha256((HERE/self.figure[key]).read_bytes()).hexdigest(),self.figure[key+'_sha256'])
        self.assertEqual(self.figure['palette'],self.reference['palette'])
        new={r['checkpoint_key']:r for r in self.figure['checkpoints']}
        for row in self.reference['checkpoints']:
            if row['pressure'] in ['h','all']:
                self.assertEqual(new[row['checkpoint_key']],row)

    def test_matched_pressure_recipe_coverage(self):
        rows=self.figure['checkpoints']
        self.assertEqual(len(rows),24)
        self.assertEqual(len({r['checkpoint_key'] for r in rows}),24)
        expected={(m,s,p,k) for m in ['14M','70M'] for s in ['4','7']
                  for p in ['h','all'] for k in [0,.05,.5]}
        self.assertEqual({(r['model'],r['scope'],r['pressure'],r['kappa']) for r in rows},expected)
        self.assertEqual(len(self.figure['panels']),6)
        for panel in self.figure['panels']:
            self.assertEqual(len(panel['checkpoint_keys']),4)
            self.assertEqual(set(panel['checkpoint_keys']),{r['checkpoint_key'] for r in rows
                if (r['model'],r['kappa'])==(panel['model'],panel['kappa'])})
        self.assertEqual({r['pressure'] for r in rows},{'h','all'})

    def test_integer_counts_and_model_denominators(self):
        index={r['checkpoint_key']:r for r in self.source['checkpoints']}
        for bar in self.figure['checkpoints']:
            row=index[bar['checkpoint_key']]; counts=row['counts']
            self.assertTrue(row['comparison_cohort'])
            for key in ['model','scope','pressure','kappa']:
                self.assertEqual(bar[key],row[key])
            self.assertEqual(set(bar['components']),set(counts['per_operation']))
            self.assertEqual(counts['model_product_count'],counts['block_product_count']+counts['lm_head_product_count'])
            for op,component in bar['components'].items():
                self.assertEqual(component['zero_product_count'],counts['per_operation'][op]['zero_product_count'])
                self.assertEqual(component['model_denominator'],counts['model_product_count'])
                self.assertGreaterEqual(component['contribution_pp'],0)
                self.assertAlmostEqual(component['contribution_pp'],100*component['zero_product_count']/component['model_denominator'],places=12)
            self.assertEqual(sum(c['zero_product_count'] for c in bar['components'].values()),counts['block_zero_product_count'])
            total=sum(c['contribution_pp'] for c in bar['components'].values())
            self.assertAlmostEqual(total,row['sparsity'],places=12)
            self.assertAlmostEqual(total,bar['total_pp'],places=12)
            self.assertLessEqual(total,self.figure['layout']['y_limits_by_model'][bar['model']][1])


if __name__=='__main__':
    unittest.main()
