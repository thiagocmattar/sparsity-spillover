"""Use a warp vote for the already-computed eight-row fallback decision."""
from io_utils import RUN, write, record


def main():
    source=RUN/'candidates/opt070'
    folder=RUN/'candidates/opt074'
    folder.mkdir(exist_ok=False)
    cu=(source/'joint.cu').read_text()
    old='''        bool done=true;
        #pragma unroll
        for(int i=0;i<8;++i)done=done && row_counts[(first_row+i)*2]<=8 && row_counts[(first_row+i)*2+1]<=8;'''
    new='''        const int2 pair=reinterpret_cast<const int2*>(row_counts)[first_row+(lane&7)];
        const bool done=__all_sync(0xffffffffu,pair.x<=8 && pair.y<=8);'''
    assert old in cu
    (folder/'joint.cu').write_text(cu.replace(old,new).rstrip()+'\n',newline='\n')
    py=(source/'joint.py').read_text().replace('run042_opt070_','run042_opt074_')
    (folder/'joint.py').write_text(py.rstrip()+'\n',newline='\n')
    code=(source/'candidate.py').read_text().replace('opt070','opt074')
    (folder/'candidate.py').write_text(code.rstrip()+'\n',newline='\n')
    write(folder/'spec.json', {'M':8,'N':256,'K':16,'short_limit':8,
          'kind':'warp-vote-fallback-decision','threads':128,
          'parent_manifest':record(RUN/'candidates/opt064/manifest.json'),
          'source_joint':record(source/'manifest.json'),
          'hypothesis':'One vector load and a warp vote replace the repeated eight-row serial completion checks; row eligibility and numerical work are unchanged.',
          'constraints':'Same actual work counters, duplicated short-row accounting, dense fallback and native model bounds.'})
    write(folder/'manifest.json', {'candidate':'opt074',
          'files':[record(p) for p in sorted(folder.glob('*'))],
          'selection_data':'fixed first16 training-development blocks only'})


if __name__=='__main__':
    main()
