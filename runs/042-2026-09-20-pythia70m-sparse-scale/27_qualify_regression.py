"""Separate an inherited synthetic rounding limit from a new kernel regression.

The original native full-model contract remains mandatory and unchanged.
No original operator failure is overwritten. Supplemental eligibility requires
bitwise identity to the already qualified frozen operator on every stress case,
exact work counters, and the same native-bound failure cases as the baseline.
"""
import sys
from io_utils import RUN,read,write,record,verify


def key(row):
    return row['case'],row['gate_h'],row['gate_z'],row['skip']


def main(identifier):
    dest=RUN/'artifacts/development'/f'operator-regression-qualification-{identifier}.json'
    if dest.exists():raise FileExistsError('Keep qualification evidence append-only')
    baseline=RUN/'artifacts/development/operator-stress-audit-opt001.json'
    candidate=RUN/'artifacts/development'/f'operator-stress-audit-{identifier}.json'
    b,c=read(baseline),read(candidate)
    assert b['status']==c['status']=='audit-complete'
    assert len(b['checks'])==len(c['checks'])==80
    assert {key(r) for r in b['checks']}=={key(r) for r in c['checks']}
    for data in (b,c):
        assert data['all_bitwise_frozen']
        assert all(row['counts']==row['oracle'] for row in data['checks'])
        assert all(row['pass'] for row in data['checks'] if row['case']!='cancellation')
        assert all(not row['frozen_passes_native_bound'] for row in data['checks'] if not row['pass'])
    assert {key(r) for r in b['checks'] if not r['pass']}=={key(r) for r in c['checks'] if not r['pass']}
    manifest=read(RUN/'candidates'/identifier/'manifest.json')
    assert c['candidate']==manifest
    for item in manifest['files']:verify(item)
    write(dest,{'status':'passed-bitwise-regression-control','candidate':identifier,
        'manifest':record(RUN/'candidates'/identifier/'manifest.json'),
        'baseline_audit':record(baseline),'candidate_audit':record(candidate),
        'script':record(__file__),
        'interpretation':'All80 outputs are bitwise equal to the retained frozen implementation, all counts match, and all72 non-cancellation native-bound cases pass. The eight added cancellation failures also occur identically in the baseline. Original failed records remain intact.',
        'final_contract':'Original native-eager logit, relative-L2 and loss bounds on every final validation block are unchanged and still required.'})
    print(identifier,'passed bitwise regression control',flush=True)


if __name__=='__main__':main(sys.argv[1])
