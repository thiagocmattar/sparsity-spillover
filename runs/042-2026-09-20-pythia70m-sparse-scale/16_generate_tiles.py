"""Mechanical M8/M16 variants; preserve per-row K order and scalar handling."""
from pathlib import Path
import subprocess
import sys
from io_utils import RUN, REPO, write, record


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError((old, text.count(old)))
    return text.replace(old, new)


def main():
    old = REPO / 'runs/040-2026-09-20-pythia70m-kernel-overhead'
    for identifier, rows, columns in [('opt001',8,256),('opt002',16,256),
                                      ('opt003',16,128),('opt004',16,512),
                                      ('opt005',8,128),('opt006',8,512)]:
        folder = RUN/'candidates'/identifier
        folder.mkdir(parents=True, exist_ok=False)
        src = old/('kernel/joint' if columns == 128 else 'candidates/opt002' if columns == 256 else 'candidates/opt003')
        cu = (src/'joint.cu').read_text()
        py = (src/('candidate.py' if columns == 128 else 'joint.py')).read_text()
        if rows == 16:
            replacements = [
                ('high=false', 'high=!hybrid || complex[local_row+8]'),
                ('hc[8],zc[8]', 'hc[16],zc[16]'),
                (f'hs[8*{columns}],zs[8*{columns}]', f'hs[16*{columns}],zs[16*{columns}]'),
                ('h_masks[8][4],z_masks[8][4]', 'h_masks[16][4],z_masks[16][4]'),
                ('first_row=blockIdx.x*8', 'first_row=blockIdx.x*16'),
                ('local<8;local+=8', 'local<16;local+=8'),
                ('i<8;++i', 'i<16;++i'),
                ('e<2;++e', 'e<4;++e'),
                ('m%8==0', 'm%16==0'),
                ('(m/8)', '(m/16)'),
                ('dim3(m/8,', 'dim3(m/16,'),
            ]
            for before, after in replacements:
                cu = replace_once(cu, before, after)
            py = replace_once(py, 'r.shape[0]//8', 'r.shape[0]//16')
        cu = '// Run042 '+identifier+f': M{rows} N{columns}.\n'+cu
        # Module name incorporates bytes, so immutable variants cannot share stale binaries.
        import re
        py = re.sub(r"name='[^']*'\+sha256\(source\)\[:12\]", "name='run042_"+identifier+"_'+sha256(source)[:12]", py)
        py = py.replace("['-O3','-lineinfo']", "['-O3','-lineinfo','--ptxas-options=-v']")
        (folder/'joint.cu').write_text(cu, newline='\n')
        (folder/'joint.py').write_text(py, newline='\n')
        candidate = f'''"""M{rows}/N{columns} h/z with native a/m and attention; unchanged gates."""
from pathlib import Path
from io_utils import RUN, module
HERE = Path(__file__).resolve().parent

def install(model):
    base = module('run042_{identifier}_base', RUN/'base70/candidate.py')
    metadata = base.install(model)
    joint = module('run042_{identifier}_joint', HERE/'joint.py')
    for layer in model.gpt_neox.layers:
        layer._run026_joint = joint.Joint(layer._run026_joint)
    return {{**metadata, 'identity':'{identifier}', 'tile_rows':{rows}, 'tile_columns':{columns},
            'sparse_sites':['h','z'], 'attention_activation_skipping':False,
            'input_projection_skipping':False, 'unchanged_accumulation_order':True}}
'''
        (folder/'candidate.py').write_text(candidate, newline='\n')
        write(folder/'spec.json', {'M':rows,'N':columns,'K':16,'origin_cuda':record(src/'joint.cu',REPO),
                                   'policy':'Same layout for both checkpoints; only h/z sparse.'})
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)
    print('Six immutable tile candidates registered')


if __name__ == '__main__': main()
