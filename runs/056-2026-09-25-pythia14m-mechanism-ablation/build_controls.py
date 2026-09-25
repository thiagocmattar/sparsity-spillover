"""Generate the minimal, checked B/S edits to the frozen K049 joint kernel."""
from pathlib import Path
from io_utils import RUN, record, write


def replace(code, old, new, count=1):
    if code.count(old) != count:
        raise ValueError(f'Frozen-source replacement mismatch: {old[:100]}')
    return code.replace(old, new)


def build_controls():
    frozen = next((RUN / 'archive/root/runs').glob('028-*'))
    source = frozen / 'candidates/k049/joint.cu'
    code = source.read_text(encoding='utf-8')
    code = replace(code, 'bool hybrid=Skip && fast_weights;',
                   'bool hybrid=Skip && fast_weights && (RUN056_TILE || RUN056_SHORT);')
    code = replace(code, 'bool h_complex=ha.count>2,z_complex=za.count>2;',
                   'bool h_complex=!RUN056_SHORT || ha.count>2,z_complex=!RUN056_SHORT || za.count>2;')
    # With S only, retain whole-group completion but never skip K16 steps in
    # a group containing any matrix-path row. Both-enabled folds to original.
    code = replace(code, 'if((h_tiles|z_tiles)==0){',
                   'if constexpr(!RUN056_TILE){if(h_tiles)h_tiles=0xffffffffu;if(z_tiles)z_tiles=0xffu;}\n'
                   '        if(RUN056_SHORT && (h_tiles|z_tiles)==0){')
    # A zero active mask in tile-only mode needs the ordinary bias/residual
    # epilogue: no row has been computed by short_linear in that mode.
    code = replace(code, 'unsigned pending=Skip?active_tiles:',
                   'unsigned pending=(Skip || hybrid)?active_tiles:')
    code = replace(code, 'accumulate<512,Skip,Count>', 'accumulate<512,Skip && RUN056_TILE,Count>')
    code = replace(code, 'accumulate<128,Skip,Count>', 'accumulate<128,Skip && RUN056_TILE,Count>')
    output = RUN / 'candidate/joint.cu'
    output.parent.mkdir(exist_ok=True)
    output.write_text(code, encoding='utf-8', newline='\n')
    write(RUN / 'provenance/control-sources.json', {
        'derivations': [{'source': record(source), 'derived': record(output)}],
        'builder': record(Path(__file__)),
        'meaning': 'Only K049 h/z changes; independent tile bypass and short-row execution.'})
