"""Publish verified calibration and the exact observed dependency lock."""
import math
from io_utils import RUN, read, write, verify, record, fs


def main():
    completed=read(RUN/'artifacts/calibration/completed.json')
    if len(completed)!=6:raise ValueError('Six calibration processes required')
    results=[read(verify(r['result'])) for r in completed]
    if not all(r['status']=='complete' and r['qualified'] and r['validation_blocks']==338 for r in results):
        raise ValueError('Calibration did not fully qualify')
    groups={}
    for name in ['p0','k050']:
        rows=[r for r in results if r['candidate']==name]
        if sorted(r['arguments']['replicate'] for r in rows)!=[1,2,3]:raise ValueError('Three fresh processes required')
        speeds=[r['timing']['candidate_graph']['paired_geomean_speedup'] for r in rows]
        groups[name]={'paired_speedups':speeds,'geomean_speedup':math.exp(sum(map(math.log,speeds))/3),
                      'wall_seconds_inside_child':[r['elapsed_seconds'] for r in rows],
                      'native_losses':[r['loss']['native'] for r in rows],
                      'candidate_losses':[r['loss']['candidate_graph'] for r in rows]}
    source=RUN/'runtime/pip-freeze.txt';dest=RUN/'provenance/pip-freeze.txt'
    # Copy an already hash-verified immutable artifact; never regenerate the lock.
    inventory=read(RUN/'artifacts/retrieval/calibration001/bundles/outputs-calibration001-inventory.json')
    verify(next(r for r in inventory['files'] if r['path']=='runtime/pip-freeze.txt'))
    data=fs(source).read_bytes()
    if fs(dest).exists() and fs(dest).read_bytes()!=data:raise ValueError('Refuse changed environment lock')
    if not fs(dest).exists():fs(dest).write_bytes(data)
    status=read(RUN/'artifacts/calibration/status.json')
    value={'qualification':'Six complete338-block passes; not part of the retrospective paper curve.',
           'condition':'c30','groups':groups,'runtime':results[0]['runtime'],
           'wall_seconds_including_lifecycle':status['elapsed_seconds'],
           'conservative_matrix_projection_hours':status['elapsed_seconds']/6*1173/3600,
           'projection_limit':'Calibration includes one diagnostic pass per six jobs; scientific matrix has35 in1173. Candidate/topology variation and retrieval require a range.',
           'maximum_allocated_bytes':max(r['peak_allocated_bytes'] for r in results),
           'dependency_lock':record(dest),'sources':[r['result'] for r in completed],
           'script':record(__file__)}
    write(RUN/'results/calibration-summary-001.json',value)
    print({k:value[k] for k in ['groups','wall_seconds_including_lifecycle','conservative_matrix_projection_hours']})


if __name__=='__main__':main()
