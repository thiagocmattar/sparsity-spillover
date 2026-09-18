"""Scientific checks for the checkpoint join and the requested derived comparisons."""
import hashlib
import json
import math
import unittest
from pathlib import Path
from paper_style import nondominated, KAPPAS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


class PaperFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((HERE/'data/paper-checkpoints.json').read_text(encoding='utf-8'))
        cls.derived=json.loads((HERE/'data/paper-derived.json').read_text(encoding='utf-8'))
        cls.index={r['checkpoint_key']:r for r in cls.data['checkpoints']}
        cls.figures={r['file']:r for r in cls.derived['figures']}

    def test_source_hashes_and_archive(self):
        self.assertEqual(hashlib.sha256((HERE/'task.md').read_bytes()).hexdigest(),self.data['task_sha256'])
        self.assertEqual(hashlib.sha256((HERE/'data/paper-checkpoints.json').read_bytes()).hexdigest(),self.derived['checkpoint_table_sha256'])
        for name,digest in self.derived['source_script_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest,name)
        for name,digest in self.data['sources_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        archived=json.loads((HERE/'figures/.archive/inventory.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(len(archived),13)
        for row in archived:
            p=HERE/'figures/.archive'/row['name']
            self.assertEqual(p.stat().st_size,row['bytes'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
            self.assertFalse((HERE/'figures'/row['name']).exists())
        self.assertEqual(len(self.derived['figures']),9)
        for row in self.derived['figures']:
            self.assertEqual(hashlib.sha256((HERE/'figures'/row['file']).read_bytes()).hexdigest(),row['sha256'])

    def test_cohort_identity_counts_and_evaluation(self):
        self.assertEqual(len(self.index),62)
        for size,n in [('14M',32),('70M',22)]:
            rows=[r for r in self.index.values() if r['model']==size and r['comparison_cohort']]
            self.assertEqual(len(rows),n)
            self.assertEqual(len({r['initial_parameter_sha256'] for r in rows}),1)
            self.assertEqual(len({r['training_schedule_hash'] for r in rows}),1)
            for scope in ['4','7']:
                for pressure in ['none','h','all'] if size=='14M' else ['h','all']:
                    self.assertEqual(sorted(r['kappa'] for r in rows if (r['scope'],r['pressure'])==(scope,pressure)),KAPPAS)
            for r in rows:
                self.assertAlmostEqual(r['sparsity'],100*r['counts']['block_zero_product_count']/r['counts']['model_product_count'])
                base=self.index[r['dense_checkpoint_key']]
                self.assertEqual(base['model'],size)
                self.assertAlmostEqual(r['dense_loss_difference'],r['loss']-base['loss'])
                self.assertAlmostEqual(r['dense_speedup'],base['latency_ms']/r['latency_ms'])
                self.assertEqual(len(r['process_latency_ms']),3)
                self.assertEqual(r['timing_precision'],'BF16')
                self.assertTrue(r['timing_device_uuid'])
                self.assertEqual(len(r['timing_indices']),64)
                self.assertAlmostEqual(r['latency_ms'],math.exp(sum(map(math.log,r['process_latency_ms']))/3),places=12)

    def test_pressure_contrasts_preserve_matching(self):
        pairs=self.figures['03-pressure-scope-threshold.pdf']['pairs']
        self.assertEqual(len(pairs),20)
        for pair in pairs:
            a,b=(self.index[pair[k]] for k in ['treatment_key','reference_key'])
            self.assertEqual((a['model'],a['scope'],a['kappa']),(b['model'],b['scope'],b['kappa']))
            self.assertEqual((a['pressure'],b['pressure']),('all','h'))
            self.assertAlmostEqual(pair['loss'],a['loss']-b['loss'])
            self.assertAlmostEqual(pair['sparsity'],a['sparsity']-b['sparsity'])
        for k,dl,ds in [(.05,.297,-.371),(.5,-.121,8.369)]:
            p=next(p for p in pairs if (p['model'],p['scope'],p['kappa'])==('70M','7',k))
            self.assertEqual(round(p['loss'],3),dl)
            self.assertEqual(round(p['sparsity'],3),ds)

    def test_operation_integer_decomposition(self):
        bars=self.figures['04-operation-sparsity-changes.pdf']['contrasts']
        self.assertEqual(len(bars),8)
        for bar in bars:
            a,b=(self.index[bar[k]] for k in ['treatment_key','reference_key'])
            delta=a['counts']['block_zero_product_count']-b['counts']['block_zero_product_count']
            self.assertEqual(sum(c['zero_product_difference'] for c in bar['components'].values()),delta)
            self.assertAlmostEqual(sum(c['contribution_pp'] for c in bar['components'].values()),a['sparsity']-b['sparsity'],places=12)

    def test_frontier_dominance_directions_and_ties(self):
        rows=[{'x':0,'y':2},{'x':1,'y':1},{'x':2,'y':2},{'x':1,'y':1}]
        self.assertEqual(nondominated(rows,'x','y'),[rows[0],rows[1],rows[3]])
        self.assertEqual(nondominated(rows,'x','y',True),[rows[1],rows[2],rows[3]])

    def test_clipping_full_coverage_and_view_limits(self):
        self.assertEqual(len(self.data['clipping']),340)
        self.assertEqual(sum(p['main_clipping'] for p in self.data['clipping']),40)
        main=self.figures['01-quality-sparsity-tradeoffs.pdf']
        lo,hi=main['y_limits']
        for size,n in [('14M',220),('70M',120)]:
            points=[p for p in self.data['clipping'] if p['model']==size]
            self.assertEqual(len(points),n)
            self.assertEqual(len(main['coverage'][size]['outside_main_y']),7)
            for key in main['coverage'][size]['trained_keys']:
                self.assertTrue(lo<=self.index[key]['dense_loss_difference']<=hi)
        for p in self.data['clipping']:
            self.assertEqual(p['coverage']['sequences'],338)
            self.assertEqual(p['coverage']['excluded_tail_tokens'],1444)

    def test_fixed_pressure_table_and_instruction_scope(self):
        self.assertEqual([r['kappa'] for r in self.derived['table2']],KAPPAS)
        for r in self.derived['table2']:
            for m in ['14M','70M']:
                a,b=(self.index[r[m][key]] for key in ['seven_key','four_key'])
                self.assertEqual((a['scope'],b['scope']),('7','4'))
                self.assertEqual((a['pressure'],b['pressure']),('h','h'))
                self.assertEqual((a['kappa'],b['kappa']),(r['kappa'],r['kappa']))
        ins=self.data['instruction']
        self.assertEqual(len(ins),30)
        for r in ins:
            self.assertEqual(self.index[r['checkpoint_key']]['model'],'14M')
            self.assertEqual(self.index[r['checkpoint_key']]['timing_session'],'Run029')
            self.assertAlmostEqual(r['projection_mma_bypass_fraction'],r['projection_mma_bypassed']/r['projection_mma_potential'])
            self.assertAlmostEqual(r['projection_scalar_zero_fraction'],r['projection_zero_products']/r['projection_products'])
            self.assertAlmostEqual(r['projection_sparse_gain'],r['all_skips_off_candidate_gm_ms']/r['projection_on_candidate_gm_ms'])


if __name__=='__main__':
    unittest.main()
