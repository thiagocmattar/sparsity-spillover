"""Verify and bundle only the seven checkpoints, validation cache and run code."""
import argparse
from pathlib import Path
import shutil
import tarfile

from density import GRID, GROUPS, read_json, sha, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', action='store_true')
    args = parser.parse_args()
    evidence = ROOT/'analyses/018-2026-09-08-results-materials/figure_data.json'
    data = read_json(evidence)
    catalog = read_json(ROOT/'runs/030-2026-09-08-all-models-posthoc-clipping/input-manifest.json')
    selected = [r for r in data['trained'] if r['scale']=='14M' and
                (r['family']=='A0' or r['family'] in ('A4-OL1','A7-OL1') and r['dose'] in (0.,.05,.5))]
    assert len(selected)==7
    rows = []
    for r in selected:
        source = next(c for c in catalog['checkpoints'] if c['id']==r['id'])
        dest = HERE/'inputs/checkpoints'/source['key']
        dest.mkdir(parents=True, exist_ok=True)
        for entry in source['files']:
            original = ROOT/source['checkpoint']/entry['name']
            assert original.stat().st_size==entry['bytes'] and sha(original)==entry['sha256']
            copy = dest/entry['name']
            shutil.copy2(original, copy)
            assert sha(copy)==entry['sha256']
        rows.append({**source, 'reference_sites':{s:r['sites'][s] for s in ('h','m','q_post','k_post','v')}})
        print('Verified',source['id'],flush=True)
    metadata_path = ROOT/'data/tokenized/minipile-pythia-14m-full/validation/metadata.json'
    metadata = read_json(metadata_path)
    tokens = metadata_path.parent/metadata['tokens_path']
    assert sha(tokens)==metadata['tokens_sha256']
    validation = HERE/'inputs/validation.bin'
    shutil.copy2(tokens, validation)
    assert sha(validation)==metadata['tokens_sha256']
    manifest = {'schema_version':1,'evidence_source':evidence.relative_to(ROOT).as_posix(),
                'evidence_sha256':sha(evidence),'checkpoints':rows,'grid':GRID,'groups':GROUPS,
                'validation':{'metadata':metadata,'bytes':validation.stat().st_size,
                              'sha256':sha(validation)},
                'evaluation':{'block_size':2048,'blocks':338,'batch_size':1,'documents':500,
                              'excluded_tail_tokens':1444,'precision':'FP32 weights; FP16 CUDA autocast',
                              'attention':'eager, uncached','posthoc_clipping':False}}
    write_json(HERE/'input-manifest.json',manifest)
    if args.bundle:
        files = [*sorted((HERE/'inputs').rglob('*')),
                 *sorted((ROOT/'src/sparsity_research').glob('*.py')),
                 *sorted(HERE.glob('*.py')), HERE/'input-manifest.json',
                 HERE/'requirements.txt', *sorted(HERE.glob('*.sh'))]
        files = [p for p in files if p.is_file()]
        inventory = {p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
        write_json(HERE/'bundle-inventory.json',inventory)
        with tarfile.open(HERE/'bundle.tar.gz','w:gz') as tar:
            for p in files+[HERE/'bundle-inventory.json']:
                tar.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
        write_json(HERE/'bundle-receipt.json',{'bytes':(HERE/'bundle.tar.gz').stat().st_size,
                   'sha256':sha(HERE/'bundle.tar.gz'),'files':len(files)+1})


if __name__=='__main__':
    main()
