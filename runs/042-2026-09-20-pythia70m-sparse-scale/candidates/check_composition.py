"""Reuse exact-source constituent tests; interactions require full-model checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from io_utils import RUN,read,write,verify,record


def main():
    folder=RUN/'candidates/opt032';spec=read(folder/'spec.json')
    manifest=read(folder/'manifest.json')
    for item in manifest['files']:verify(item)
    evidence={}
    for component,identifier in spec['parents'].items():
        parent=read(verify(spec['parent_manifests'][component]))
        for item in parent['files']:verify(item)
        test=RUN/'artifacts/development'/f'operator-{identifier}.json'
        if test.exists() and read(test)['status']=='passed':
            assert read(test)['candidate']==parent
        else:
            test=RUN/'artifacts/development'/f'operator-regression-qualification-{identifier}.json'
            result=read(test)
            assert result['status']=='passed-bitwise-regression-control'
            assert read(verify(result['manifest']))==parent
            for name in ('baseline_audit','candidate_audit'):verify(result[name])
        evidence[component]=record(test)
    write(RUN/'artifacts/development/operator-opt032.json',{
        'status':'passed','candidate':manifest,'checker':record(__file__),
        'kind':'Exact-source reuse of previously tested constituent operators',
        'constituent_evidence':evidence,
        'limits':'This source-identity check does not establish interaction correctness; full-model development and final native-eager comparisons remain required.'})


if __name__=='__main__':main()
