"""Recover complete, inventory-hashed files from interrupted archive downloads.

Identical content already retrieved from another condition may fill a missing
file only when its byte count and SHA-256 match the destination's own inventory.
This does not mark an incomplete condition verified or delete cloud storage.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import yaml
from run_config import condition_specs, load_config, resolved_condition_config

HERE=Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def save_member(archive, member, expected=None):
    target=(HERE/member.name).resolve()
    if not target.is_relative_to((HERE/'artifacts').resolve()) or not member.isfile():
        raise ValueError('Unexpected archive member: '+member.name)
    if expected and member.size!=expected['bytes']:raise ValueError('Size mismatch')
    source=archive.extractfile(member)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        digest=hashlib.file_digest(source,'sha256').hexdigest()
        if sha(target)!=digest:raise ValueError('Existing artifact differs: '+str(target))
    else:
        with target.open('xb') as out:shutil.copyfileobj(source,out)
    if expected and sha(target)!=expected['sha256']:raise ValueError('Hash mismatch: '+str(target))
    return target

def main():
    attempts={}
    for evidence in sorted((HERE/'prelaunch/cloud').iterdir()):
        metadata=evidence/'metadata.tar.gz'
        if not metadata.exists():continue
        expected=json.loads((evidence/'metadata-sha256.json').read_text())['sha256']
        assert sha(metadata)==expected
        with tarfile.open(metadata) as archive:
            inventory_member=next(m for m in archive.getmembers() if m.name.endswith('/transfer_inventory.json'))
            inventory=json.load(archive.extractfile(inventory_member))
            attempt_rel=Path(inventory_member.name).parent
            lookup={str(attempt_rel/row['path']).replace('\\','/'):row for row in inventory['files']}
            for member in archive.getmembers():
                save_member(archive,member,lookup.get(member.name))
        attempts[evidence.name]={'dir':HERE/attempt_rel,'inventory':inventory,'lookup':lookup}
    recovered=[]
    for pod, info in attempts.items():
        path=HERE/'retrieval'/pod/'results.tar'
        if not path.exists():continue
        try:
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.offset_data+member.size>path.stat().st_size:break
                    if member.isfile() and member.name in info['lookup']:
                        target=save_member(archive,member,info['lookup'][member.name])
                        recovered.append({'path':str(target.relative_to(HERE)).replace('\\','/'),'source_pod':pod})
        except tarfile.ReadError:
            pass # Complete preceding members are individually verified above.
    content={}
    for info in attempts.values():
        for row in info['inventory']['files']:
            path=info['dir']/row['path']
            if path.exists():
                assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
                content.setdefault((row['bytes'],row['sha256']),path)
    reused=[]
    regenerated=[]
    config=load_config()
    by_id={row['id']:row for row in condition_specs(config)}
    summaries=[]
    for pod,info in attempts.items():
        missing=[]
        for row in info['inventory']['files']:
            path=info['dir']/row['path']
            if not path.exists():
                if row['path']=='config.yaml':
                    manifest=json.loads((info['dir']/'manifest.json').read_text())
                    value=resolved_condition_config(config,by_id[manifest['condition']['id']])
                    data=yaml.safe_dump(dict(value),sort_keys=False).encode('utf-8')
                    if len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']:
                        with path.open('xb') as out:out.write(data)
                        regenerated.append({'destination':str(path.relative_to(HERE)).replace('\\','/'),'sha256':row['sha256'],'method':'Reviewed frozen config and original resolved-condition serializer; exact original inventory hash matched'})
                        continue
                source=content.get((row['bytes'],row['sha256']))
                if source:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with source.open('rb') as inp,path.open('xb') as out:shutil.copyfileobj(inp,out)
                    assert sha(path)==row['sha256']
                    reused.append({'source':str(source.relative_to(HERE)).replace('\\','/'),'destination':str(path.relative_to(HERE)).replace('\\','/'),'sha256':row['sha256']})
                else:missing.append(row)
        summary={'pod':pod,'attempt':info['dir'].name,'missing_files':missing,'missing_bytes':sum(row['bytes'] for row in missing),'inventory_complete':not missing}
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    previous=HERE/'prelaunch/salvage-receipt.json'
    old=json.loads(previous.read_text()) if previous.exists() else {}
    receipt={'method':'Per-file byte counts and SHA-256 against each condition original inventory; full archive hashes remain unavailable','recovered_members':recovered,'identical_content_copies':old.get('identical_content_copies',[])+reused,'regenerated_exact_hash_configs':old.get('regenerated_exact_hash_configs',[])+regenerated,'conditions':summaries}
    (HERE/'prelaunch/salvage-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
