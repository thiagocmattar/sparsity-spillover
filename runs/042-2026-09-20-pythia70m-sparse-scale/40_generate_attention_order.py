"""Schedule long causal query blocks across all heads before shorter blocks."""
import shutil
from io_utils import RUN, read, write, record


def main():
    source=RUN/'candidates/opt064'
    folder=RUN/'candidates/opt073'
    shutil.copytree(source/'attention',folder/'attention')
    py=(folder/'attention/candidate.py').read_text().replace('run042_opt064_','run042_opt073_')
    (folder/'attention/candidate.py').write_text(py.rstrip()+'\n',newline='\n')
    cu=(folder/'attention/kernel.cu').read_text()
    assert 'dim3(32,1,8)' in cu
    cu=cu.replace('dim3(32,1,8)','dim3(8,1,32)')
    (folder/'attention/kernel.cu').write_text(cu.rstrip()+'\n',newline='\n')
    header=(folder/'attention/flash_fwd_kernel.h').read_text()
    start=header.index('    const int m_block = gridDim.x - 1 - blockIdx.x;')
    end=header.index('    const int bidh = blockIdx.z;',start)+len('    const int bidh = blockIdx.z;')
    before=header[start:end]
    after=before.replace('gridDim.x - 1 - blockIdx.x','gridDim.z - 1 - blockIdx.z').replace('bidh = blockIdx.z','bidh = blockIdx.x')
    header=header[:start]+after+header[end:]
    (folder/'attention/flash_fwd_kernel.h').write_text(header.rstrip()+'\n',newline='\n')
    code=(source/'candidate.py').read_text().replace('opt064','opt073')
    code=code.replace('Output layout only; unchanged dense attention arithmetic.','Token-major output and head-first launch ordering; unchanged dense attention arithmetic.')
    (folder/'candidate.py').write_text(code.rstrip()+'\n',newline='\n')
    write(folder/'spec.json', {'kind':'dense-attention-head-first-order',
          'parent_manifest':record(RUN/'candidates/opt063/manifest.json'),
          'source_attention':record(source/'manifest.json'),
          'hypothesis':'Distribute long causal query blocks across heads before shorter blocks, reducing the final scheduling tail.',
          'qualification':'Only CTA order changes; same grids, arithmetic, causal mask, outputs and original development bounds.'})
    write(folder/'manifest.json', {'candidate':'opt073',
          'files':[record(p) for p in sorted(folder.rglob('*')) if p.is_file()],
          'selection_data':'fixed first16 training-development blocks only'})


if __name__=='__main__':
    main()
