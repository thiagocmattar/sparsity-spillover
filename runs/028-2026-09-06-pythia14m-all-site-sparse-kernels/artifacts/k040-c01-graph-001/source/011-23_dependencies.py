"""Fetch immutable upstream headers and licenses for the native-Flash fork.

Unmodified dependencies live in ignored runtime storage; every file and source
archive is inventoried. Flash matches Torch's pinned Flash submodule; CUTLASS
matches that Flash submodule's own dependency, not Torch's separate CUTLASS pin.
No model, cache or numerical setting is changed by this preparation step.
"""
import argparse
import shutil
import tarfile
import urllib.request
from common import RUN,ROOT,record,read_json,write_json,verify_record,sha256

SOURCES={
    'flash':('Dao-AILab/flash-attention','e2743ab5b3803bb672b16437ba98a3b1d4576c50','csrc/flash_attn/src/'),
    'cutlass':('NVIDIA/cutlass','7127592069c2fe01b041e174ba4345ef9b279671','include/'),
}
ARCHIVE_HASHES={'flash':'edfc138477144f15b51ae12e3dbf942bfe452f2eb5c6583c324a2873e630d44b',
                'cutlass':'12baa9c2bf22eaca065b5e55387c21bb4ac742992a6ecd2dd1d3b016dfd25a0a'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    destination=RUN/'runtime/vendor';destination.mkdir(parents=True,exist_ok=True)
    inventory=destination/'inventory.json'
    if inventory.exists():
        for row in read_json(inventory)['files']:verify_record(row)
        print({'status':'verified','files':len(read_json(inventory)['files'])});return
    if a.verify:raise ValueError('Dependency inventory absent')
    files=[];sources=[]
    for name,(repository,commit,prefix) in SOURCES.items():
        url=f'https://codeload.github.com/{repository}/tar.gz/{commit}'
        archive=RUN/'bundles'/f'upstream-{name}-{commit}.tar.gz'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            partial=archive.with_suffix('.partial')
            with urllib.request.urlopen(url,timeout=90) as remote,partial.open('wb') as local:shutil.copyfileobj(remote,local)
            partial.replace(archive)
        if sha256(archive)!=ARCHIVE_HASHES[name]:raise ValueError('Pinned upstream archive hash mismatch')
        root=destination/name;root.mkdir(parents=True,exist_ok=True)
        selected=[]
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                if not member.isfile():continue
                short=member.name.split('/',1)[1]
                wanted=short.startswith(prefix) or short in {'LICENSE','LICENSE.txt','AUTHORS','AUTHORS.txt'}
                if not wanted:continue
                target=(root/short).resolve()
                if not target.is_relative_to(root.resolve()):raise ValueError('Archive path escapes dependency root')
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                selected.append(record(target))
        if not selected or not (root/'LICENSE').exists() and not (root/'LICENSE.txt').exists():raise ValueError('Missing source/license')
        files+=selected;sources.append({'name':name,'repository':repository,'commit':commit,'url':url,'archive':record(archive),'files':len(selected)})
        print({'dependency':name,'files':len(selected),'archive':record(archive)},flush=True)
    write_json(inventory,{'sources':sources,'files':files,'bytes':sum(r['bytes'] for r in files)})
    for row in files:verify_record(row)
    print({'status':'verified','files':len(files),'bytes':sum(r['bytes'] for r in files)},flush=True)

if __name__=='__main__':main()
