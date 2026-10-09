"""Mathematical checks for the frontier and OL1 normalizations used here."""
import importlib.util
import unittest
from evidence import ROOT, frontier

spec=importlib.util.spec_from_file_location('geometry_collector',ROOT/'analyses/043-2026-10-09-global-tile-head-frontier/02_collect_geometry.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


class AnalysisMath(unittest.TestCase):
    def test_frontier_requires_both_objectives(self):
        points=[dict(id=n,loss=l,latency_ms=t) for n,l,t in
                [('quality',1,4),('middle',2,2),('speed',4,1),('worse',3,3),('same_loss_slower',2,3)]]
        self.assertEqual([r['id'] for r in frontier(points)],['quality','middle','speed'])

    def test_equal_coordinates_preserve_both_executions(self):
        points=[dict(id=n,loss=1,latency_ms=1) for n in ['session-a','session-b']]
        self.assertEqual(len(frontier(points)),2)

    def test_cohort_membership_can_differ_from_global_membership(self):
        body=dict(id='body',loss=1,latency_ms=1)
        head=dict(id='head',loss=2,latency_ms=2)
        self.assertEqual(frontier([body,head]),[body])
        self.assertEqual(frontier([head]),[head])

    def test_removed_component_uses_two_distinct_denominators(self):
        # u=(2,0), w=(-4,3): removed component is approximately (4,0).
        row=dict(task_pressure_dot_before=-8,task_direction_norm=2)
        alpha=collector.geometry.opposing_component_ratio(row)
        self.assertAlmostEqual(100*alpha,200)  # manuscript task normalization may exceed 100%
        self.assertAlmostEqual(100*alpha*2/5,80)  # pressure norm fraction
        # Pressure energy fraction would instead be 64%, so do not label the norm as energy.
        self.assertAlmostEqual((alpha*2/5)**2*100,64)

    def test_nonopposing_direction_removes_nothing(self):
        row=dict(task_pressure_dot_before=8,task_direction_norm=2)
        self.assertEqual(collector.geometry.opposing_component_ratio(row),0)


if __name__=='__main__':unittest.main()
