"""Freeze only after both-endpoint full-model training checks; no final peeking."""
import argparse
from support import RUN,read,write,sha,source_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--attempts',nargs=2,required=True);a=p.parse_args()
    selection=read(RUN/'provenance'/a.selection)
    if not selection['changed_sites']:raise ValueError('No component winner: return to development')
    ids=set();sources=[]
    for attempt in a.attempts:
        root=RUN/'artifacts'/attempt;r=read(root/'result.json');q=read(root/'quality.json');t=read(root/'timing.json')
        assert r['arguments']['phase']=='training' and r['validation_blocks']==128
        ids.add(r['arguments']['condition'])
        assert r['selection']['sha256']==sha(RUN/'provenance'/a.selection)
        for mode in ('candidate_graph','prior_c_graph','native_graph','native_hz_graph','dense_fused_graph','dense_policy_graph','candidate_h_off_graph','candidate_z_off_graph','candidate_hz_off_graph'):
            assert q['pass'][mode],(attempt,mode)
        for split in ('development','confirmation'):
            for mode in ('candidate_graph','prior_c_graph','dense_policy_graph'):
                assert q['split_gate_pass'][split][mode],(attempt,split,mode)
        # Both endpoints must improve over prior C and optimized dense, even if
        # a larger neighbour difference could be manufactured by a regression.
        def latency(mode):return t['summary'][mode]['geomean_host_ms']
        assert latency('candidate_graph')<min(latency(mode) for mode in ('prior_c_graph','native_graph','native_hz_graph','dense_fused_graph','dense_policy_graph'))
        for row in r['source_hashes'].items():assert sha(RUN/row[0])==row[1],row[0]
        sources.append({'path':str((root/'result.json').relative_to(RUN)),'sha256':sha(root/'result.json')})
    assert ids=={'c24','c25'}
    selection['training_qualification']=sources
    selection['kernel_source_hashes']=source_hashes()
    target=RUN/'provenance/selection-final.json'
    if target.exists():raise FileExistsError(target)
    write(target,selection);print('Frozen after two qualified full-model training evaluations')


if __name__=='__main__':main()
