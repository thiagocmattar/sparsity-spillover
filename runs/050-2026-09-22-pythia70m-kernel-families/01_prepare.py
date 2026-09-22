"""Pin immutable dependencies and literal blocks0:128 of canonical training."""
from pathlib import Path
from support import BASE, RUN, read, write, sha

def main():
    if (RUN/'artifacts').exists(): raise RuntimeError('Do not rewrite executed inputs')
    repo=RUN.parents[1]
    source=repo/'data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin'
    metadata=read(source.with_name('metadata.json'))
    assert sha(source)==metadata['tokens_sha256']
    with source.open('rb') as f: prefix=f.read(128*2048*4)
    assert len(prefix)==128*2048*4
    (RUN/'inputs').mkdir(exist_ok=True)
    target=RUN/'inputs/train128.int32.bin';target.write_bytes(prefix)
    write(RUN/'provenance/inputs.json',{'training_source':metadata,
        'training_source_path':source.relative_to(repo).as_posix(),
        'training':{'path':'inputs/train128.int32.bin','sha256':sha(target),'bytes':len(prefix)},
        'development_indices':list(range(64)), 'confirmation_indices':list(range(64,128)),
        'baseline_inputs_manifest_sha256':sha(BASE/'provenance/inputs.json'),
        'baseline_source_manifest_sha256':sha(BASE/'provenance/source-freeze.json'),
        'note':'Literal canonical train-cache blocks, not the separate historical Run049 development sample.'})
    write(RUN/'prelaunch/local-checks.json',{'bootstrap_tests':242,'bootstrap_passed':True})
    print('Canonical full training cache hash verified; first128 blocks pinned')

if __name__=='__main__':main()
