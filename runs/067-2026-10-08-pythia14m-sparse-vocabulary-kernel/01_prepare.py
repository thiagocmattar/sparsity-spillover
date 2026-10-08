"""Seal the four retained endpoints, native gate runtime and validation bytes."""
import argparse
import io
import tarfile
from io_utils import REPO, RUN, SOURCE, read, record, write, verify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle', action='store_true')
    args=parser.parse_args()
    comparison=read(SOURCE/'results/comparison.json')
    rows=[r for r in comparison['rows'] if r['kind']!='control']
    if len(rows)!=4 or {(r['kappa'],tuple(r['pressure_sites'])) for r in rows}!={(k,p) for k in (0.,.1) for p in (('h',),('h','head_in'))}:
        raise ValueError('Expected all four completed Run066 conditions')
    sources=[REPO/'src/sparsity_research'/p.name for p in (REPO/'src/sparsity_research').glob('*.py')]
    sources += [SOURCE/n for n in ('topology.py','tile_gate.py','head_site.py','results/comparison.json','config.yaml')]
    checkpoints=[]
    for row in rows:
        attempt=SOURCE/row['source']
        checkpoint=attempt/'checkpoints/step_000712'
        files=[record(checkpoint/n) for n in ('config.json','generation_config.json','model.safetensors','checkpoint_metadata.json')]
        config=read(checkpoint/'config.json')
        expected={'operator':'tile_corner_threshold','kappa':row['kappa'],'tile_tokens':8,'tile_features':16}
        if config['head_input_gate']!=expected or any(config['site_gates'][s]!=expected for s in ('h','z')):
            raise ValueError('Checkpoint gate identity mismatch')
        if (config['hidden_size'],config['intermediate_size'],config['num_hidden_layers'],config['vocab_size'])!=(128,512,6,50304):
            raise ValueError('Checkpoint architecture mismatch')
        evidence=[record(attempt/n) for n in ('manifest.json','metrics.json','config.yaml')]
        checkpoints.append({'id':row['id'],'checkpoint':checkpoint.relative_to(REPO).as_posix(),
                            'kappa':row['kappa'],'pressure_sites':row['pressure_sites'],
                            'production_loss':row['loss'],'files':files,'provenance':evidence})
    cache=REPO/'data/tokenized/minipile-pythia-14m-full/validation'
    validation=record(cache/'tokens.int32.bin')
    if validation['bytes']!=693668*4 or validation['sha256']!='51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451':
        raise ValueError('Full validation byte identity mismatch')
    inputs={'source_run':SOURCE.name,'checkpoints':checkpoints,'validation':validation,
            'validation_metadata':record(cache/'metadata.json'), 'runtime_dependencies':[record(p) for p in sorted(sources)],
            'initial_parameter_sha256':'ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57',
            'training_schedule_sha256':'f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa',
            'training_seeds':{'model':1234,'data_order':1234},'new_training':False}
    write(RUN/'provenance/inputs.json',inputs)
    all_files=inputs['runtime_dependencies']+[validation,inputs['validation_metadata']]
    all_files += [r for c in checkpoints for r in c['files']+c['provenance']]
    all_files += [record(p) for p in sorted(RUN.glob('*.py')) if p.name!='test_run067.py']
    all_files += [record(RUN/n) for n in ('head.cu','config.json','provenance/inputs.json','00_setup_remote.sh','03_execute.sh')]
    all_files=sorted({r['path']:r for r in all_files}.values(),key=lambda r:r['path'])
    for row in all_files:verify(row)
    write(RUN/'provenance/upload-inventory.json',all_files)
    if args.bundle:
        destination=REPO/'tmp/run067-upload.tar.gz'
        destination.parent.mkdir(exist_ok=True)
        with tarfile.open(destination,'w:gz') as archive:
            for row in all_files:archive.add(REPO/row['path'],arcname=row['path'],recursive=False)
            archive.add(RUN/'provenance/upload-inventory.json',arcname=(RUN/'provenance/upload-inventory.json').relative_to(REPO).as_posix())
        write(RUN/'prelaunch/bundle.json',{'archive':record(destination),'members':len(all_files)+1,
                                         'uncompressed_bytes':sum(r['bytes'] for r in all_files)})
    print({'checkpoints':len(checkpoints),'verified_files':len(all_files),'validation_tokens':693668})


if __name__=='__main__':main()
