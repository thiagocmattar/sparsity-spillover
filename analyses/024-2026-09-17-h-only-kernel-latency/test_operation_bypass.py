"""Count pooling and retained-evidence contracts for operation bypass."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('operation_bypass', HERE/'18_reduce_operation_bypass.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OperationBypassTests(unittest.TestCase):
    def test_count_first_pooling(self):
        result = module.pooled_bypass([1,90], [9,0])
        self.assertEqual(result['potential_mmas'],100)
        self.assertEqual(result['bypass_fraction'],.09)
        self.assertNotEqual(result['bypass_fraction'],.45)

    def test_full_bypass_and_invalid_counters(self):
        self.assertEqual(module.pooled_bypass([0,0],[2,3])['bypass_fraction'],1)
        for issued,bypassed in [([0],[0]),([-1],[2]),([1.0],[2]),([1],[2,3]),([],[])]:
            with self.assertRaises(ValueError):module.pooled_bypass(issued,bypassed)

    def test_complete_scientific_coverage_and_identities(self):
        data=json.loads((HERE/'data/operation-bypass.json').read_text())
        rows=data['settings']
        self.assertEqual(len({r['setting_id'] for r in rows}),102)
        for size,n in [('14M',40),('70M',22)]:
            trained=[r for r in rows if r['model']==size and r['kind']=='trained']
            self.assertEqual(len({r['checkpoint_key'] for r in trained}),n)
        for size in ['14M','70M']:
            for family in ['A0','A1-H']:
                clips=[r for r in rows if r['model']==size and r['kind']=='clipping' and r['family']==family]
                self.assertEqual(sorted(r['p'] for r in clips),[i/10 for i in range(10)])
                self.assertEqual(len({r['checkpoint_key'] for r in clips}),1)
        for row in rows:
            self.assertEqual(set(row['operations']),set(module.OPS))
            for op in row['operations'].values():
                self.assertEqual(op['issued_mmas']+op['bypassed_mmas'],op['potential_mmas'])
                self.assertEqual(op['bypass_fraction'],op['bypassed_mmas']/op['potential_mmas'])

    def test_control_padding_and_scalar_substitution_remain_visible(self):
        data=json.loads((HERE/'data/operation-bypass.json').read_text())
        rows={r['setting_id']:r for r in data['settings']}
        for key in ['trained-14M-c01','trained-70M-c00']:
            pv=rows[key]['operations']['probability_value']
            self.assertGreater(pv['bypass_fraction'],.05)
            self.assertEqual(pv['bypass_difference_from_base_pp'],0)
        high=rows['trained-14M-c30']
        self.assertGreater(high['operations']['qk_scores']['bypass_fraction'],.5)
        self.assertGreater(high['operations']['probability_value']['bypass_fraction'],.6)
        self.assertGreater(high['operations']['mlp_w2']['simt_products'],0)
        self.assertLess(high['skip_ablation']['attention_gain'],1)
        self.assertEqual(sum(r['skip_ablation'] is not None for r in rows.values()),35)


if __name__ == '__main__':unittest.main()
