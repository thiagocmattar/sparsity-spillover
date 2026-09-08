"""Publish the verified signed histograms and audit their trained gate regions."""
from pathlib import Path
import shutil
import numpy as np

from density import edges, read_json, sha, write_json

HERE = Path(__file__).resolve().parent


def main():
    retrieved = HERE/'artifacts/retrieved'
    receipt = read_json(HERE/'transfer-receipt.json')
    for name, entry in receipt['inventory'].items():
        path = retrieved/name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    out = HERE/'results'
    (out/'histograms').mkdir(parents=True,exist_ok=True)
    full = retrieved/'attempts/002-runpod-full'
    manifest = read_json(full/'manifest.json')
    assert manifest['status']=='completed' and not manifest['smoke']
    assert manifest['completed_blocks']==2366
    for path in full.glob('*.json.gz'):
        shutil.copy2(path,out/'histograms'/path.name)
    for name in ('manifest.json','verification.json','events.jsonl'):
        shutil.copy2(full/name,out/name)
    for name in ('calibration.json','cuda-boundary-checks.json','environment.txt',
                 'gpu.txt','check_cuda.py','setup.log'):
        shutil.copy2(retrieved/name,out/name)
    grid = edges()
    report = []
    for path in sorted((out/'histograms').glob('*.gz')):
        data = read_json(path)
        source = data['source']
        kappa = source['training_parameter']
        comparison = data['reference_comparison']
        checks = []
        if source['family']!='A0':
            for row in data['rows']:
                site = row['name'].split('.layer_')[0]
                counts = np.asarray(row['histogram'],dtype=np.int64)
                if site in ('h','m'):
                    forbidden = grid[1:] < kappa-.001
                    assert counts[forbidden].sum()==0 and row['underflow']==0
                    checks.append(row['name'])
                elif source['family']=='A7-OL1' and kappa>0:
                    forbidden = (grid[:-1] > -kappa+.001) & (grid[1:] < kappa-.001)
                    assert counts[forbidden].sum()==0
                    checks.append(row['name'])
        report.append({'key':source['key'],'loss':data['coverage']['loss'],
            'loss_delta':comparison['loss_delta'],
            'max_site_zero_fraction_delta':max(abs(s['exact_zero_fraction_delta']) for s in comparison['sites'].values()),
            'max_site_rms_delta':max(abs(s['rms_delta']) for s in comparison['sites'].values()),
            'empty_gate_region_checks':checks})
    verification = {'source_script':'08_consolidate.py','checkpoints':report,
        'all_complete_blocks':2366,'all_documents_per_checkpoint':500,
        'histogram_partition_and_pooling':'passed remote and local 03_verify.py',
        'gate_boundary_margin':.001,
        'gate_checks_note':'Native bins wholly inside each forbidden region; one bin margin avoids FP16 boundary rounding.'}
    write_json(out/'scientific-verification.json',verification)
    inventory = {p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('inventory.json','README.md')}
    write_json(out/'inventory.json',inventory)
    print(f'Published {len(report)} full histograms; {sum(len(r["empty_gate_region_checks"]) for r in report)} gate-region checks passed')


if __name__=='__main__':
    main()
