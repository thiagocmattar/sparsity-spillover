"""Create two bounded N-tile candidates from the unchanged h/z implementation."""
from pathlib import Path
import ast

RUN = Path(__file__).resolve().parent.parent


def make(identifier, width):
    folder = RUN/'candidates'/identifier
    folder.mkdir(exist_ok=False)
    source = (RUN/'kernel/joint/joint.cu').read_text()
    blocks, atoms = 512//width, width//64
    def change(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, (old, source.count(old), count)
        source = source.replace(old, new)
    change('((blockIdx.x*4+blockIdx.y)*8+warp)*6', f'((blockIdx.x*{blocks}+blockIdx.y)*8+warp)*6')
    change('hs[8*128],zs[8*128]', f'hs[8*{width}],zs[8*{width}]')
    change('ha.count*128', f'ha.count*{width}')
    change('za.count*128', f'za.count*{width}')
    change('for(int e=0;e<4;++e)', f'for(int e=0;e<{width//32};++e)')
    change('blockIdx.y*128+lane*4+e,slot=local*128+lane*4+e',
           f'blockIdx.y*{width}+lane*{width//32}+e,slot=local*{width}+lane*{width//32}+e')
    change('warp<4?512:0,0,warp<4?128:0', f'warp<8?{128*atoms}:0,0,warp<8?{32*atoms}:0')
    change('if(warp>=4)', 'if(warp>=8)')
    change('blockIdx.y*128+warp*32', f'blockIdx.y*{width}+warp*{width//8}')
    change('slot=lr*128+c%128', f'slot=lr*{width}+c%{width}')
    change('(m/8)*4*8*6', f'(m/8)*{blocks}*8*6')
    change('dim3(m/8,4)', f'dim3(m/8,{blocks})')
    if atoms != 4:
        change('float (&acc)[4][4]', f'float (&acc)[{atoms}][4]')
        change('for(int atom=0;atom<4;++atom)', f'for(int atom=0;atom<{atoms};++atom)', 2)
        change('bypassed+=(32-__popc(pending))*4', f'bypassed+=(32-__popc(pending))*{atoms}')
        change('bypassed+=4', f'bypassed+={atoms}')
        change('issued+=4', f'issued+={atoms}')
        change('float ah[4][4]={},az[4][4]={}', f'float ah[{atoms}][4]={{}},az[{atoms}][4]={{}}')
    source = f'// Run040 {identifier}: N{width}, eight compute warps; same M8 arithmetic and gate policy.\n'+source
    (folder/'joint.cu').write_text(source, newline='\n')
    py = (RUN/'kernel/joint/candidate.py').read_text().split('\ndef install(')[0]
    py = py.replace("'run035_joint_'", f"'run040_{identifier}_joint_'")
    assert '(r.shape[0]//8,4,8,6)' in py
    py = py.replace('(r.shape[0]//8,4,8,6)', f'(r.shape[0]//8,{blocks},8,6)')
    (folder/'joint.py').write_text(py, newline='\n')
    entry = f'''"""Native a/m and attention, with N{width} h/z output tiles."""
from pathlib import Path
from io_utils import RUN, module
from controls import original_norms, substitute

HERE = Path(__file__).resolve().parent

def install(model):
    saved = original_norms(model)
    frozen = module('run040_{identifier}_frozen', RUN/'kernel/candidate.py')
    metadata = frozen.install(model, shortcut=False, round_p=False, skip=True, projection_skip=True)
    for mode in ('native-am', 'native-attention'):
        substitute(model, mode, saved)
    joint = module('run040_{identifier}_joint', HERE/'joint.py')
    for layer in model.gpt_neox.layers:
        layer._run026_joint = joint.Joint(layer._run026_joint)
    return {{**metadata, 'identity': 'run040-{identifier}', 'new_optimization_search': True,
            'input_projections': 'Native PyTorch linear with existing fused gates',
            'attention': 'Native causal SDPA with existing gated Q/K/V operands',
            'output_projections': 'M8 N{width}, eight compute warps; same short-row policy, K16 masks, and accumulation order',
            'output_projection_counter_limits': 'Unchanged M8 padded-M16 potential and scalar-product accounting; N-grid resized only',
            'dispatch': 'Same component and tile policy for both checkpoints'}}
'''
    ast.parse(entry); ast.parse(py)
    (folder/'candidate.py').write_text(entry, newline='\n')
    (folder/'README.md').write_text(f'''# {identifier}: N{width} h/z output tiles

Keep opt001's native a/m and attention replacements. Expand each h/z output
tile from128 to{width} columns, using all eight existing warps for matrix
work. Each eight-row activation group is consequently inspected{blocks}
time(s), instead of four, across the512 output columns. This tests redundant
inspection and output scheduling; reduced inspection requests are not a claim
about physical DRAM traffic or measured speedup.

The16-row MMA instructions still contain eight real rows. Preserve the exact
short-row test, gates, K16 support masks, accumulation order, branch/residual
rounding, and total potential-MMA/scalar counter definitions. More columns
per block can increase register pressure or reduce parallelism; those are
measured risks, not assumed improvements. One policy applies to both checkpoints.

The T7/Ph initial trace measured about0.20ms in h/z. Qualify operators and both
checkpoints on the fixed16 training-development blocks. No optimization
validation evaluation or timing is used to choose this candidate.
''', newline='\n')


if __name__ == '__main__':
    make('opt002', 256)
    make('opt003', 512)
