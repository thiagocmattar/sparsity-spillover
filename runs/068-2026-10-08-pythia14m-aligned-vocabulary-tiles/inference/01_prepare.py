"""Seal terminal trained endpoints or explicitly labeled development models."""
import argparse
import tarfile
from io_utils import RUN, REPO, SOURCE, read, record, verify, write
import sys
sys.path.insert(0,str(SOURCE))
from run_config import load_config,condition_specs,resolved_condition_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args();cfg=load_config();checkpoints=[]
    if not args.development:
        verification=read(SOURCE/'artifacts/verification.json')
        if verification['status']!='verified' or verification['condition_count']!=2:
            raise ValueError('Both verified terminal conditions required')
    for condition in condition_specs(cfg):
        if args.development:
            checkpoint=SOURCE/'prelaunch/local-checkpoints'/condition['id']
            evidence=[record(SOURCE/'prelaunch/local-smoke.json')];loss=None
        else:
            attempts=[p for p in (SOURCE/'artifacts/attempts').iterdir() if p.is_dir()
                      and read(p/'metrics.json')['condition']['id']==condition['id']]
            if len(attempts)!=1:raise ValueError('One verified attempt per condition required')
            attempt=attempts[0];checkpoint=attempt/'checkpoints/step_000712'
            evidence=[record(attempt/n) for n in ('manifest.json','metrics.json','config.yaml')]
            loss=read(attempt/'metrics.json')['validation']['step_000712']['loss']
        actual=read(checkpoint/'config.json');expected=resolved_condition_config(cfg,condition)['model']
        if actual['head_input_gate']!=expected['head_input_gate'] or actual['site_gates']!=expected['site_gates']:
            raise ValueError('Independent block/head gate mismatch')
        if (actual['hidden_size'],actual['intermediate_size'],actual['num_hidden_layers'],actual['vocab_size'])!=(128,512,6,50304):
            raise ValueError('Architecture mismatch')
        names=['config.json','generation_config.json','model.safetensors']
        if not args.development:names.append('checkpoint_metadata.json')
        checkpoints.append({'id':condition['id'],'checkpoint':checkpoint.relative_to(REPO).as_posix(),
            'kappa':condition['head_gate_threshold'],'block_kappa':condition['block_gate']['kappa'],
            'pressure_sites':condition['pressure_sites'],'production_loss':loss,
            'files':[record(checkpoint/n) for n in names],'provenance':evidence,'development_model':args.development})
    cache=REPO/'data/tokenized/minipile-pythia-14m-full/validation'
    validation=record(cache/'tokens.int32.bin')
    if validation['bytes']!=693668*4 or validation['sha256']!='51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451':
        raise ValueError('Full validation byte mismatch')
    dependencies=list((REPO/'src/sparsity_research').glob('*.py'))+list(SOURCE.glob('*.py'))
    dependencies += [SOURCE/'config.yaml',SOURCE/'inputs/random-initialization/provenance.json',SOURCE/'inputs/random-initialization/model.safetensors']
    dependencies += [SOURCE.parent/'004-2026-08-29-pythia14m-full-pass-l1n'/n for n in
        ('run_config.py','initialization.py','optimizer_boundary.py','diagnostics.py','06_build_cache_from_hf.py')]
    inputs={'source_run':SOURCE.name,'checkpoints':checkpoints,'validation':validation,
        'validation_metadata':record(cache/'metadata.json'),
        'runtime_dependencies':[record(p) for p in sorted(dependencies) if not p.name.startswith('test_')],
        'initial_parameter_sha256':cfg['comparison']['initial_parameter_sha256'],
        'training_schedule_sha256':cfg['comparison']['training_schedule_sha256'],
        'training_seeds':cfg['seeds'],'new_training':True,'development':args.development}
    prefix='' if args.development else 'trained-'
    input_path=RUN/'provenance'/f'{prefix}inputs.json'
    inventory_path=RUN/'provenance'/f'{prefix}upload-inventory.json'
    if not args.development and (input_path.exists() or inventory_path.exists()):
        raise FileExistsError('Preserve sealed terminal endpoint identities')
    write(input_path,inputs)
    rows=inputs['runtime_dependencies']+[validation,inputs['validation_metadata']]
    rows += [r for c in checkpoints for r in c['files']+c['provenance']]
    rows += [record(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')]
    rows += [record(RUN/n) for n in ('head.cu','config.json','00_setup_remote.sh','03_execute.sh')]+[record(input_path)]
    rows=sorted({r['path']:r for r in rows}.values(),key=lambda r:r['path'])
    for row in rows:verify(row)
    write(inventory_path,rows)
    if args.bundle:
        destination=REPO/'tmp/run068-head-upload.tar.gz'
        with tarfile.open(destination,'w:gz') as archive:
            for row in rows:archive.add(REPO/row['path'],arcname=row['path'],recursive=False)
            archive.add(inventory_path,arcname=inventory_path.relative_to(REPO).as_posix())
        write(RUN/'prelaunch/bundle.json',{'archive':record(destination),'members':len(rows)+1})
    print({'models':len(checkpoints),'verified_files':len(rows),'development':args.development},flush=True)


if __name__=='__main__':main()
