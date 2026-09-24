"""Collect the six verified trained endpoints and their 18 qualified timing processes."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/054-2026-09-24-pythia31m-t2-ph'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def gm(values):
    return math.exp(math.fsum(map(math.log,values))/len(values))


def main():
    verification = RUN/'artifacts/verification.json'
    cohort = read(verification)
    assert cohort['status']=='verified' and len(cohort['conditions'])==6
    grids = [p for p in (RUN/'latency/artifacts').glob('final-*-grid.json')]
    source_hashes = {verification.relative_to(ROOT).as_posix():sha(verification)}
    points = []
    for training in cohort['conditions']:
        condition = training['condition']
        matches = [p for p in grids if read(p)['condition']==condition['id']]
        if len(matches)!=1:
            raise ValueError('Require exactly one final grid for '+condition['id'])
        grid_path = matches[0]; grid = read(grid_path)
        assert grid['status']=='completed' and len(grid['replicates'])==3
        source_hashes[grid_path.relative_to(ROOT).as_posix()] = sha(grid_path)
        attempt = RUN/'artifacts/attempts'/training['attempt']
        final = attempt/training['final_checkpoint']['path']
        metrics = read(attempt/'metrics.json')
        coverage = metrics['validation']['final']
        assert coverage['sequences']==338 and coverage['excluded_tail_tokens']==1444
        assert coverage['input_tokens']==692224 and math.isfinite(coverage['loss'])
        latencies = {mode:[] for mode in ('kernel_graph','native_graph')}
        processes = []
        for replica in grid['replicates']:
            assert replica['returncode']==0 and replica['qualified']
            folder = RUN/'latency/artifacts/attempts'/replica['attempt']
            manifest, quality, timing = [read(folder/name) for name in ('manifest.json','quality.json','timing.json')]
            assert manifest['status']=='completed' and manifest['qualified'] and manifest['scientific_measurement']
            assert manifest['arguments']['condition']==condition['id']
            assert manifest['runtime']['gpu']=='NVIDIA GeForce RTX 5090'
            assert manifest['config']['precision']=='bfloat16' and manifest['implementation']['identity']=='opt073-31m-v1'
            assert quality['blocks']==338 and quality['excluded_tail_tokens']==1444 and all(quality['pass'].values())
            if manifest['arguments']['replicate']==1:
                diagnostics_path=folder/'runtime-diagnostics.json';diagnostics=read(diagnostics_path)
                assert diagnostics['blocks']==338 and diagnostics['input_tokens']==692224
                assert diagnostics['documents']==500 and diagnostics['excluded_tail_tokens']==1444
                assert len(diagnostics['activation_rows'])==42
                assert len(diagnostics['h_z_work'])==len(diagnostics['row_nnz_histograms'])==12
                source_hashes[diagnostics_path.relative_to(ROOT).as_posix()]=sha(diagnostics_path)
            for record in manifest['checkpoint_files']:
                path = final/record['path']
                assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
            for name in ('manifest.json','quality.json','timing.json'):
                path = folder/name;source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            for mode in latencies:
                values = [r['host_ms'] for r in timing['samples'] if r['mode']==mode]
                assert len(values)==448 and all(math.isfinite(x) and x>0 for x in values)
                value = gm(values)
                assert math.isclose(value,timing['summary'][mode]['geomean_host_ms'],rel_tol=1e-12)
                latencies[mode].append(value)
            processes.append(dict(attempt=replica['attempt'],replicate=manifest['arguments']['replicate'],
                gpu_uuid=manifest['runtime']['gpu_uuid'],losses=quality['loss'],source_identity=manifest['source_identity']['content_sha256']))
        assert sorted(p['replicate'] for p in processes)==[1,2,3]
        assert len({p['gpu_uuid'] for p in processes})==1
        assert len({p['source_identity'] for p in processes})==1
        points.append(dict(model='31M',parameters=30494720,scope='0' if condition['is_control'] else 'hz',
            pressure='none' if condition['is_control'] else 'h',kappa=condition['gate_threshold'],condition=condition,
            loss=coverage['loss'],coverage=coverage,R_model=training['R_model'],R_block=training['R_block'],
            checkpoint_key=sha(final/'model.safetensors'),source_attempt=attempt.relative_to(ROOT).as_posix(),
            latency_ms=gm(latencies['kernel_graph']),implementation_latency_ms={k:gm(v) for k,v in latencies.items()},
            latency_process_ms=latencies,processes=processes,kernel='opt073-31m-v1',timing_session='Run054',
            qualified=True))
    assert sorted(p['kappa'] for p in points if p['scope']=='hz')==[0.,.01,.05,.1,.5]
    assert len({p['checkpoint_key'] for p in points})==6
    assert len({r['source_identity'] for p in points for r in p['processes']})==1
    assert len({r['gpu_uuid'] for p in points for r in p['processes']})==1
    output = HERE/'data/31m-results.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(points=points,sources_sha256=source_hashes),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps([dict(condition=p['condition']['id'],loss=p['loss'],latency_ms=p['implementation_latency_ms']) for p in points],indent=2))


if __name__=='__main__':
    main()
