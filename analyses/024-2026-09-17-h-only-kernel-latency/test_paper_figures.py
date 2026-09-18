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
        self.assertEqual({(p['model'],p['scope'],p['kappa']) for p in pairs},
                         {(m,s,k) for m in ['14M','70M'] for s in ['4','7'] for k in KAPPAS})
        self.assertEqual(len({p[key] for p in pairs for key in ['treatment_key','reference_key']}),40)
        for pair in pairs:
            a,b=(self.index[pair[k]] for k in ['treatment_key','reference_key'])
            self.assertEqual((a['model'],a['scope'],a['kappa']),(b['model'],b['scope'],b['kappa']))
            self.assertEqual((a['pressure'],b['pressure']),('all','h'))
            self.assertEqual((pair['treatment'],pair['reference']),('all','h'))
            self.assertEqual(a['initial_parameter_sha256'],b['initial_parameter_sha256'])
            self.assertEqual(a['training_schedule_hash'],b['training_schedule_hash'])
            self.assertAlmostEqual(pair['loss'],a['loss']-b['loss'])
            self.assertAlmostEqual(pair['latency_us'],1000*(a['latency_ms']-b['latency_ms']),places=12)
            self.assertEqual(pair['sessions'],[a['timing_session'],b['timing_session']])
            self.assertEqual(a['timing_workload'],b['timing_workload'])
            self.assertEqual(a['timing_indices'],b['timing_indices'])
            # T7 at 14M is the retained cross-session comparison, not a paired timing sample.
            cross_session=(pair['model'],pair['scope'])==('14M','7')
            self.assertEqual(a['timing_session']!=b['timing_session'],cross_session)
            self.assertEqual(a['counts']['model_product_count'],b['counts']['model_product_count'])
            expected=100*(a['counts']['block_zero_product_count']-b['counts']['block_zero_product_count'])/a['counts']['model_product_count']
            self.assertAlmostEqual(pair['sparsity'],expected,places=12)
        # The difference of the two P0-referenced effects must recover all-minus-h.
        previous={ (p['scope'],p['kappa'],p['pressure']):p
                   for p in self.figures['A2-pressure-versus-none.pdf']['pairs'] }
        for p in pairs:
            if p['model']!='14M':
                continue
            a,b=(previous[p['scope'],p['kappa'],pressure] for pressure in ['all','h'])
            self.assertEqual((p['treatment_key'],p['reference_key']),
                             (a['treatment_key'],b['treatment_key']))
            self.assertAlmostEqual(p['loss'],a['loss']-b['loss'])
            self.assertAlmostEqual(p['sparsity'],a['sparsity']-b['sparsity'])
            self.assertAlmostEqual(p['latency_us'],a['latency_us']-b['latency_us'],places=10)

    def test_operation_integer_decomposition(self):
        bars=self.figures['04-operation-sparsity-changes.pdf']['checkpoints']
        self.assertEqual(len(bars),12)
        self.assertEqual(len({b['checkpoint_key'] for b in bars}),12)
        self.assertEqual({(b['model'],b['scope'],b['pressure'],b['kappa']) for b in bars},
                         {('14M',s,p,k) for s in ['4','7'] for p in ['none','h','all'] for k in [.05,.5]})
        for bar in bars:
            row=self.index[bar['checkpoint_key']]
            self.assertTrue(row['comparison_cohort'])
            self.assertEqual({k:bar[k] for k in ['model','scope','pressure','kappa']},
                             {k:row[k] for k in ['model','scope','pressure','kappa']})
            self.assertEqual(set(bar['components']),set(row['counts']['per_operation']))
            for op,c in bar['components'].items():
                self.assertEqual(c['zero_product_count'],row['counts']['per_operation'][op]['zero_product_count'])
                self.assertEqual(c['model_denominator'],row['counts']['model_product_count'])
                self.assertGreaterEqual(c['contribution_pp'],0)
                self.assertAlmostEqual(c['contribution_pp'],100*c['zero_product_count']/c['model_denominator'],places=12)
            self.assertEqual(sum(c['zero_product_count'] for c in bar['components'].values()),row['counts']['block_zero_product_count'])
            self.assertAlmostEqual(sum(c['contribution_pp'] for c in bar['components'].values()),row['sparsity'],places=12)
            self.assertAlmostEqual(bar['total_pp'],row['sparsity'],places=12)

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
