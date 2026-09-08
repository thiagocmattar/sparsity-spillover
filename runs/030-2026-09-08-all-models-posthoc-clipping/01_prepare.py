"""Describe the exact manuscript checkpoints and verify inference inputs locally."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    evidence_path = ROOT/'analyses/018-2026-09-08-results-materials/figure_data.json'
    data = json.loads(evidence_path.read_text())
    rows = []
    for r in data['trained']:
        attempt = ROOT/r['source']
        manifest = json.loads((attempt/'manifest.json').read_text())
        final = manifest['checkpoints']['final']
        checkpoint = attempt/final['path']
        inventory = json.loads((attempt/'diagnostics/checkpoint_inventory.json').read_text())
        needed = not (r['scale']=='14M' and r['family'] in
                      ('A0','A1-H','A1-H-L1','A1-H-OL1','A4')) and r['family'] not in ('A0','A1-H')
        files = []
        for p in sorted(checkpoint.iterdir()):
            if p.name == 'training_state.pt':
                continue
            entry = next(x for x in inventory['files'] if x['path']==checkpoint.name+'/'+p.name)
            assert p.stat().st_size==entry['bytes']
            if needed:
                assert sha(p)==entry['sha256'], str(p)
            files.append({'name':p.name,'bytes':entry['bytes'],'sha256':entry['sha256']})
        rows.append({'id':r['id'],'key':r['id'].replace(':','_'),
                     'scale':r['scale'],'family':r['family'],'training_parameter':r['dose'],
                     'evaluate':needed,'source':r['source'],
                     'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
                     'checkpoint_content_sha256':final['content_sha256'],
                     'manifest_sha256':sha(attempt/'manifest.json'),
                     'inventory_sha256':sha(attempt/'diagnostics/checkpoint_inventory.json'),
                     'files':files,'topology':manifest['topology'],
                     'source_loss':r['loss'],'source_R_model':r['R_model'],
                     'identity':r['identity'],'ceiling':r['ceiling']})
        print(r['id'], 'verified for evaluation' if needed else 'existing sweep reused',flush=True)
    assert len(rows)==54 and sum(r['evaluate'] for r in rows)==35
    cache = ROOT/'data/tokenized/minipile-pythia-14m-full'
    target = HERE/'inputs'
    target.mkdir(exist_ok=True)
    identities = {}
    for split in ('train','validation'):
        meta = json.loads((cache/split/'metadata.json').read_text())
        token_path = cache/split/meta['tokens_path']
        assert sha(token_path)==meta['tokens_sha256']
        with token_path.open('rb') as f:
            content = f.read(10*2048*4) if split=='train' else f.read()
        dest = target/(split+'.bin')
        dest.write_bytes(content)
        identities[split]={'source_metadata':meta,'source_tokens_sha256':meta['tokens_sha256'],
                           'transferred_bytes':len(content),'transferred_sha256':sha(dest),
                           'selection':'first ten complete source-order blocks' if split=='train' else 'all source tokens'}
    result = {'schema_version':1,'cohort_source':evidence_path.relative_to(ROOT).as_posix(),
              'cohort_source_sha256':sha(evidence_path),'checkpoints':rows,'caches':identities,
              'targets':[i/10 for i in range(10)],'clipping_sites':['a','m','h','z'],
              'new_points':350,'reused_points':190,'total_points':540}
    (HERE/'input-manifest.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')


if __name__=='__main__':
    main()
