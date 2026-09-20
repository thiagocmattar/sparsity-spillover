"""Retain executed variants; correct actual scalar work in a new immutable candidate."""
from io_utils import RUN, write, record


def main():
    source = RUN/'candidates/opt051'
    folder = RUN/'candidates/opt063'
    folder.mkdir(exist_ok=False)
    cu = (source/'joint.cu').read_text()
    needle = '    counters<Count>(stats,warp,lane,hi,hb,zi,zb,h_scalar,z_scalar);'
    assert cu.count(needle) == 1
    cu = cu.replace(needle, '''    // The parallel prepass also computes short rows in a mixed fallback group.
    // Those outputs are recomputed below the fallback path; count both executions.
    if constexpr(Count){
        if(row_counts && row_counts[(first_row+warp)*2]<=8 && row_counts[(first_row+warp)*2+1]<=8){
            h_scalar+=row_counts[(first_row+warp)*2]*256;
            z_scalar+=row_counts[(first_row+warp)*2+1]*256;
        }
    }
''' + needle)
    (folder/'joint.cu').write_text(cu,newline='\n')
    py = (source/'joint.py').read_text().replace('run042_opt051_', 'run042_opt063_')
    py = py.replace('self.short_limit=8', 'self.short_limit=8;self.parallel_prepass_duplicates=True')
    assert 'parallel_prepass_duplicates=True' in py
    (folder/'joint.py').write_text(py.rstrip()+'\n',newline='\n')
    code = (source/'candidate.py').read_text().replace('opt051','opt063')
    code = code.replace('exact issued MMA and scalar counts.', 'actual issued MMA and scalar counts, including duplicate prepass work in mixed fallback groups.')
    (folder/'candidate.py').write_text(code,newline='\n')
    write(folder/'spec.json', {'M':8,'N':256,'K':16,'short_limit':8,
          'kind':'parallel-row-inspection-actual-counts','threads':128,
          'parent_manifest':record(RUN/'candidates/opt032/manifest.json'),
          'source_joint':record(source/'manifest.json'),
          'reason':'Correct opt051 scalar-work accounting: mixed fallback groups recompute prepass outputs for short rows. Timed count=False execution and outputs are unchanged.',
          'qualification':'New operator audit includes an independent mixed-group duplicate-work oracle. Previous records remain unchanged.'})
    write(folder/'manifest.json', {'candidate':'opt063',
          'files':[record(p) for p in sorted(folder.glob('*'))],
          'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
