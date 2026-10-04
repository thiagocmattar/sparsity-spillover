"""Stage verified scalar controls now; add final tile checkpoints after training."""
import argparse
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify

SCALAR = REPO / 'runs/033-2026-09-17-a7-h-only-k050-benchmark'
TRAIN = RUN.parent
BASE = REPO / 'runs/061-2026-10-04-pythia14m-corner-tile-latency'


def copy(source, relative):
    target = RUN / relative
    fs(target.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fs(source), fs(target))
    assert record(source, REPO)['sha256'] == record(target)['sha256']
    return record(target)


def scalar():
    if (RUN / 'artifacts/attempts').exists():
        raise RuntimeError('Cannot change benchmark inputs after execution begins')
    old = read(SCALAR / 'provenance/inputs.json')
    selected = [row for row in old['checkpoints'] if row['dose'] in (0., .1)]
    assert {row['id'] for row in selected} == {'c36', 'c39'}
    checkpoints = []
    for row in selected:
        for entry in row['files'] + row['provenance']:
            verify(entry, SCALAR)
        files = [copy(SCALAR / entry['path'], entry['path']) for entry in row['files']]
        provenance = [copy(SCALAR / entry['path'], entry['path']) for entry in row['provenance']]
        checkpoints.append({**row, 'family': 'scalar', 'files': files, 'provenance': provenance,
                            'source': row['original_checkpoint'],
                            'architecture': row['canonical_logical_products']['architecture_maximum']})
    previous = read(BASE / 'provenance/inputs.json')
    verify(previous['validation'], BASE)
    validation = copy(BASE / previous['validation']['path'], 'inputs/validation.int32.bin')
    write(RUN / 'provenance/scalar-inputs.json', {
        'checkpoints': checkpoints, 'validation': validation,
        'validation_metadata': previous['validation_metadata'],
        'scalar_source_manifest': record(SCALAR / 'provenance/inputs.json', REPO)})
    print('Verified and staged c36/c39 scalar T7/Ph controls and full validation')


def final():
    if (RUN / 'artifacts/attempts').exists():
        raise RuntimeError('Benchmark inputs are immutable after execution begins')
    if read(TRAIN / 'artifacts/verification.json')['status'] != 'verified':
        raise ValueError('Both training conditions must pass full verification')
    inputs = read(RUN / 'provenance/scalar-inputs.json')
    rows = list(inputs['checkpoints'])
    for source in sorted((TRAIN / 'artifacts/attempts').iterdir()):
        if not (source / 'manifest.json').is_file():
            continue
        manifest = read(source / 'manifest.json')
        condition = manifest['condition']
        if manifest['status'] != 'completed' or manifest['completed_steps'] != 712:
            raise ValueError('Incomplete tile checkpoint')
        if condition['active_sites'] != ['a','m','h','q_post','k_post','v','z'] or condition['pressure_sites'] != ['h']:
            raise ValueError('Wrong tile topology or pressure')
        identifier = condition['id']
        checkpoint = source / 'checkpoints/step_000712'
        names = ['config.json', 'model.safetensors', 'checkpoint_metadata.json', 'generation_config.json']
        files = [copy(checkpoint / name, f'inputs/checkpoints/{identifier}/{name}') for name in names]
        provenance = [copy(source / name, f'provenance/originals/{identifier}/{name}')
                      for name in ['manifest.json','config.yaml','diagnostics/logical_products.json']]
        rows.append({'id': identifier, 'family': 'tile', 'dose': condition['gate_threshold'],
            'checkpoint': f'inputs/checkpoints/{identifier}', 'files': files, 'provenance': provenance,
            'source': source.relative_to(REPO).as_posix(), 'source_condition': condition,
            'original_checkpoint': checkpoint.relative_to(REPO).as_posix(),
            'original_files': [record(checkpoint / name, REPO) for name in names],
            'final_checkpoint_content_sha256': manifest['checkpoints']['final']['content_sha256'],
            'architecture': read(source / 'diagnostics/logical_products.json')['architecture_maximum']})
    expected = {identifier for pair in read(RUN / 'config.json')['pair_conditions'].values() for identifier in pair}
    if len(rows) != 4 or {row['id'] for row in rows} != expected:
        raise ValueError('Exactly the approved four final T7/Ph checkpoints required')
    inputs['checkpoints'] = rows
    write(RUN / 'provenance/inputs.json', inputs)
    print('Four final T7/Ph checkpoints staged')


def development():
    """Untrained tile models for integration only; never benchmark endpoints."""
    import yaml
    import torch
    from safetensors.torch import load_file
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from topology import register
    register()
    from sparsity_research.pythia import apply_activation_topology
    provenance = read(TRAIN / 'inputs/random-initialization/provenance.json')
    assert provenance['trained_updates'] == 0
    seed_config = read(REPO / provenance['source'] / 'config.json')
    config = yaml.safe_load((TRAIN / 'config.yaml').read_text())
    state = load_file(TRAIN / 'inputs/random-initialization/model.safetensors')
    inputs = read(RUN / 'provenance/scalar-inputs.json')
    rows = list(inputs['checkpoints'])
    torch.set_num_threads(1)
    for kappa in [0., .1]:
        token = f'{kappa:g}'.replace('.', 'p')
        identifier = f't7-tile-ph-kappa-{token}'
        architecture = GPTNeoXConfig(**seed_config)
        architecture.topology_id = 'A7-Z-POST'
        architecture.site_gate = None
        architecture.site_gates = {site: {**spec, 'kappa': kappa} for site, spec in config['model']['site_gates'].items()}
        model = apply_activation_topology(GPTNeoXForCausalLM(architecture), torch=torch)
        model.load_state_dict(state, strict=True)
        checkpoint = RUN / 'inputs/development-checkpoints' / identifier
        model.save_pretrained(checkpoint, safe_serialization=True)
        rows.append({'id': identifier, 'family': 'untrained_development_tile', 'dose': kappa,
                     'checkpoint': checkpoint.relative_to(RUN).as_posix(),
                     'files': [record(path) for path in checkpoint.iterdir() if path.is_file()],
                     'trained_updates': 0})
    inputs['checkpoints'] = rows
    write(RUN / 'provenance/development-inputs.json', inputs)
    print('Prepared two explicitly untrained development tile checkpoints')


def bundle(tag):
    if not tag.isalnum():
        raise ValueError('Simple unique tag required')
    inputs = read(RUN / 'provenance/inputs.json')
    rows = [inputs['validation']]
    rows += [entry['snapshot'] for entry in read(RUN / 'provenance/archive.json')['files']]
    rows += [entry for checkpoint in inputs['checkpoints'] for entry in checkpoint['files'] + checkpoint['provenance']]
    for row in rows:
        verify(row)
    paths = [RUN / row['path'] for row in rows]
    paths += list(RUN.glob('*.py')) + list(RUN.glob('*.cu')) + list(RUN.glob('*.sh')) + [RUN / 'config.json']
    paths += [path for path in (RUN / 'provenance').glob('*.json') if path.name != 'development-inputs.json']
    paths += [RUN / 'provenance/pip-freeze.txt']
    paths = sorted(set(paths))
    output = RUN / 'bundles' / f'input-{tag}.tar'
    fs(output.parent).mkdir(exist_ok=True)
    if fs(output).exists():
        raise FileExistsError(output)
    write(RUN / 'bundles' / f'input-{tag}-inventory.json', {'files': [record(path) for path in paths]})
    with tarfile.open(fs(output), 'w') as archive:
        for path in paths:
            archive.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(output.with_suffix('.receipt.json'), record(output))
    print(record(output))


def verify_all():
    inputs = read(RUN / 'provenance/inputs.json')
    rows = [inputs['validation']]
    rows += [entry['snapshot'] for entry in read(RUN / 'provenance/archive.json')['files']]
    rows += [entry for checkpoint in inputs['checkpoints'] for entry in checkpoint['files'] + checkpoint['provenance']]
    for row in rows:
        verify(row)
    print('Verified', len(rows), 'frozen input/source files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['scalar','final','bundle','development','verify'])
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    {'scalar': scalar, 'final': final, 'bundle': lambda: bundle(args.tag), 'development': development, 'verify': verify_all}[args.action]()
