"""Freeze Run029 c30 inputs and prepare narrowly modified execution controls."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

BASE = REPO / 'runs/029-2026-09-07-pythia14m-matched-kernel-retrospective'


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f'Frozen-source replacement mismatch: {old[:90]}')
    return text.replace(old, new)


def prepare():
    if (RUN / 'artifacts/attempts').exists():
        raise RuntimeError('Inputs are immutable once execution begins')
    copies = []

    def copy(source, relative, expected=None):
        if expected and record(source, BASE) != expected:
            raise ValueError(f'Source identity mismatch: {source}')
        dest = RUN / relative
        if not dest.resolve().is_relative_to(RUN):
            raise ValueError('Copy escapes run')
        fs(dest.parent).mkdir(parents=True, exist_ok=True)
        if not fs(dest).exists() or sha(dest) != sha(source):
            shutil.copyfile(fs(source), fs(dest))
        copies.append({'source': record(source, REPO), 'copy': record(dest)})

    archived = read(BASE / 'provenance/archive.json')
    for row in archived['files']:
        item = row['snapshot']
        copy(BASE / item['path'], item['path'], item)
    write(RUN / 'provenance/archive.json', archived)
    original = read(BASE / 'provenance/inputs.json')
    endpoint = next(r for r in original['checkpoints'] if r['id'] == 'c30')
    for row in endpoint['files'] + endpoint['provenance'] + [original['validation']]:
        copy(BASE / row['path'], row['path'], row)
    weight = next(r for r in endpoint['files'] if r['path'].endswith('/model.safetensors'))
    assert weight['sha256'] == 'f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425'
    write(RUN / 'provenance/inputs.json', {**original, 'checkpoints': [endpoint]})
    copy(BASE / 'provenance/pip-freeze.txt', 'provenance/pip-freeze.txt')
    catalog = read(BASE / 'provenance/candidates.json')
    catalog['configurations'] = [r for r in catalog['configurations'] if r['id'] == 'k050']
    assert len(catalog['configurations']) == 1
    assert catalog['configurations'][0]['settings']['shortcut'] is False
    write(RUN / 'provenance/candidates.json', catalog)
    write(RUN / 'provenance/reuse.json', {'base_run': BASE.relative_to(REPO).as_posix(), 'files': copies})
    build_controls()
    verify_inputs()


def build_controls():
    frozen = next((RUN / 'archive/root/runs').glob('028-*'))
    out = RUN / 'candidate'
    out.mkdir(exist_ok=True)
    changes = []

    def save(source, name, text):
        dest = out / name
        dest.write_text(text, encoding='utf-8', newline='\n')
        changes.append({'source': record(source), 'derived': record(dest)})

    # Compile-time controls retain the original interface, layout and fusions.
    source = frozen / 'candidates/k049/joint.cu'
    code = source.read_text(encoding='utf-8')
    code = replace(code, 'bool hybrid=Skip && fast_weights;',
                   'bool hybrid_h=Skip && fast_weights && RUN037_SKIP_H;'
                   'bool hybrid_z=Skip && fast_weights && RUN037_SKIP_Z;bool hybrid=hybrid_h || hybrid_z;')
    code = replace(code, 'auto ha=inspect<512>(h,row,lane,th,gh);auto za=inspect<128>(z,row,lane,tz,gz);',
                   'TinyRow ha{},za{};if(hybrid_h)ha=inspect<512>(h,row,lane,th,gh);'
                   'if(hybrid_z)za=inspect<128>(z,row,lane,tz,gz);')
    code = replace(code, 'bool h_complex=ha.count>2,z_complex=za.count>2;',
                   'bool h_complex=!hybrid_h || ha.count>2,z_complex=!hybrid_z || za.count>2;')
    code = replace(code, 'h_complex?ha.tiles:0', 'h_complex?(hybrid_h?ha.tiles:0xffffffffu):0')
    code = replace(code, 'z_complex?za.tiles:0', 'z_complex?(hybrid_z?za.tiles:0xffu):0')
    code = replace(code, 'accumulate<512,Skip,Count>', 'accumulate<512,Skip && RUN037_SKIP_H,Count>')
    code = replace(code, 'accumulate<128,Skip,Count>', 'accumulate<128,Skip && RUN037_SKIP_Z,Count>')
    code = replace(code, 'th,gh,hybrid,hc', 'th,gh,hybrid_h,hc')
    code = replace(code, 'tz,gz,hybrid,zc', 'tz,gz,hybrid_z,zc')
    code = replace(code, 'hybrid && !hc[lr]', 'hybrid_h && !hc[lr]')
    code = replace(code, 'hybrid && !zc[lr]', 'hybrid_z && !zc[lr]')
    save(source, 'joint.cu', code)
    for name in ['kernel.cu', 'sparse_gemm.h', 'flash_fwd_kernel.h']:
        source = frozen / 'candidates/k035' / name
        code = source.read_text(encoding='utf-8')
        if name == 'flash_fwd_kernel.h':
            code = replace(code, 'run028_gemm<Kernel_traits::Run028Skip,',
                           'run028_gemm<(Kernel_traits::Run028Skip && RUN037_SKIP_QK),', 2)
            code = replace(code, 'run028_gemm_rs<Kernel_traits::Run028Skip,',
                           'run028_gemm_rs<(Kernel_traits::Run028Skip && RUN037_SKIP_PV),', 2)
        save(source, name, code)
    source = frozen / 'runtime/vendor/flash/LICENSE'
    save(source, 'LICENSE-FLASH', source.read_text(encoding='utf-8'))
    source = frozen / '115_hybrid_diagnostics.py'
    code = replace(source.read_text(encoding='utf-8'),
                   'expected=hybrid_counts(value,original.fast_weights,original.skip)',
                   "expected=hybrid_counts(value,original.fast_weights,getattr(original,'skip_'+site,original.skip))")
    code = replace(code, 'row=projection_counts(value,n)',
                   "row=projection_counts(value,n)\n"
                   "                        if site in {'a','m'}:\n"
                   "                            layer=model.gpt_neox.layers[int(name.rsplit('_',1)[1])]\n"
                   "                            linear=layer.attention.query_key_value if site=='a' else layer.mlp.dense_h_to_4h\n"
                   "                            if not linear._run028_projection.skip:\n"
                   "                                row['issued_mmas']+=row['skipped_mmas'];row['skipped_mmas']=0")
    dest = RUN / 'diagnostics.py'
    dest.write_text(code, encoding='utf-8', newline='\n')
    changes.append({'source': record(source), 'derived': record(dest)})
    write(RUN / 'provenance/control-sources.json', {'derivations': changes, 'builder': record(Path(__file__))})


def verify_inputs():
    manifest = read(RUN / 'provenance/inputs.json')
    rows = [manifest['validation']]
    rows += [r['snapshot'] for r in read(RUN / 'provenance/archive.json')['files']]
    rows += [f for c in manifest['checkpoints'] for f in c['files'] + c['provenance']]
    for row in rows:
        verify(row)
    for row in read(RUN / 'provenance/control-sources.json')['derivations']:
        verify(row['source']); verify(row['derived'])
    print(f'Verified {len(rows)} frozen source/input identities and derived control sources')


def bundle(tag):
    verify_inputs()
    if not tag.isalnum():
        raise ValueError('Alphanumeric bundle tag required')
    target = RUN / 'bundles' / f'input-{tag}.tar'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError('Use a new bundle tag')
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN / 'config.json']
    paths += list((RUN / 'candidate').iterdir()) + list((RUN / 'provenance').glob('*'))
    paths += [RUN / r['copy']['path'] for r in read(RUN / 'provenance/reuse.json')['files']]
    paths = sorted({p for p in paths if fs(p).is_file()})
    write(target.with_name(f'input-{tag}-inventory.json'), {'files': [record(p) for p in paths]})
    with tarfile.open(target, 'w') as tar:
        for path in paths:
            tar.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(target.with_suffix('.receipt.json'), record(target))
    print(json.dumps(record(target)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'verify', 'bundle'])
    p.add_argument('--tag', default='001')
    args = p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'verify': verify_inputs()
    else: bundle(args.tag)
