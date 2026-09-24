"""Verify six completed conditions, retaining explicitly recorded infrastructure failures."""
import importlib.util
import json
from pathlib import Path
import sys

RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(RUN))
from run_config import load_config,condition_specs,write_json


def main():
    spec=importlib.util.spec_from_file_location('run054_verification',RUN/'03_verify.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    retired=RUN/'prelaunch/retired-training-attempts.json'
    excluded=json.loads(retired.read_text())['attempts'] if retired.exists() else []
    excluded_by_id={r['attempt']:r for r in excluded}
    results=[];found_excluded=[];config=load_config()
    for attempt in sorted((RUN/'artifacts/attempts').glob('*')):
        if not attempt.is_dir():continue
        manifest=json.loads((attempt/'manifest.json').read_text())
        if attempt.name in excluded_by_id:
            retirement=excluded_by_id[attempt.name]
            pipeline=json.loads((RUN/'artifacts/pipeline'/retirement['pipeline']/'status.json').read_text())
            receipt=json.loads((RUN/'prelaunch'/('retrieved-'+retirement['pipeline']+'.json')).read_text())
            if pipeline['status']!='failed' or receipt['status']!='verified' or not receipt['retired']:
                raise ValueError('Only terminal, retrieved infrastructure failures may be excluded')
            if retirement['signal']!='SIGTERM' or manifest['status'] not in ('running','failed'):
                raise ValueError('Unexpected termination record or overwritten manifest')
            found_excluded.append(excluded_by_id[attempt.name]);continue
        results.append(verifier.verify(attempt,config))
    if len(results)!=6 or {r['condition']['id'] for r in results}!={r['id'] for r in condition_specs(config)}:
        raise ValueError('Exactly one complete verified result per approved condition required')
    if {r['attempt'] for r in found_excluded}!=set(excluded_by_id):
        raise ValueError('An excluded infrastructure attempt was not retrieved')
    (RUN/'artifacts/verification.json').write_text(json.dumps(dict(status='verified',conditions=results,
        excluded_infrastructure_attempts=found_excluded),indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Six conditions verified; infrastructure-failure evidence retained')


if __name__=='__main__':main()
