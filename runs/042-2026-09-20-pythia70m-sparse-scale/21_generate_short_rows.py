"""Extend scalar execution to4/8/16 nonzeros; retain original output bounds."""
import subprocess
import sys
from io_utils import RUN,read,write,record


def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def main():
    for identifier,origin,limit in [('opt015','opt001',4),('opt016','opt001',8),
                                    ('opt017','opt001',16),('opt018','opt002',8)]:
        source=RUN/'candidates'/origin;folder=RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False);spec=read(source/'spec.json')
        cu=(source/'joint.cu').read_text()
        cu=once(cu,'struct TinyRow { int count=3,i0=0,i1=0; float v0=0.f,v1=0.f; unsigned tiles[4]={}; };',
                f'struct TinyRow {{ int count={limit+1},indices[{limit}]={{}}; float values[{limit}]={{}}; unsigned tiles[4]={{}}; }};')
        cu=once(cu,'if(out.count+__popc(mask)>2)out.count=3;',
                f'if(out.count+__popc(mask)>{limit})out.count={limit+1};')
        cu=once(cu,'out.count=3;break;',f'out.count={limit+1};break;')
        cu=once(cu,'if(out.count==0){out.i0=base+selected;out.v0=next;}\n            else{out.i1=base+selected;out.v1=next;}',
                'out.indices[out.count]=base+selected;out.values[out.count]=next;')
        cu=once(cu,'if(a.count>0)value=__fmul_rn(a.v0,__bfloat162float(wt[a.i0*512+col]));\n    if(a.count==2)value=__fmaf_rn(a.v1,__bfloat162float(wt[a.i1*512+col]),value);',
                'if(a.count>0)value=__fmul_rn(a.values[0],__bfloat162float(wt[a.indices[0]*512+col]));\n'
                f'    #pragma unroll\n    for(int i=1;i<{limit};++i)if(i<a.count)value=__fmaf_rn(a.values[i],__bfloat162float(wt[a.indices[i]*512+col]),value);')
        cu=once(cu,'bool h_complex=ha.count>2,z_complex=za.count>2;',
                f'bool h_complex=ha.count>{limit},z_complex=za.count>{limit};')
        (folder/'joint.cu').write_text(cu,newline='\n')
        py=(source/'joint.py').read_text().replace('run042_'+origin,'run042_'+identifier)
        py=once(py,'self.skip=previous.skip;self.count=False;',f'self.skip=previous.skip;self.short_limit={limit};self.count=False;')
        (folder/'joint.py').write_text(py,newline='\n')
        candidate=(source/'candidate.py').read_text().replace(origin,identifier)
        candidate=candidate.replace("'unchanged_accumulation_order':True",f"'unchanged_accumulation_order':False,'short_row_limit':{limit}")
        (folder/'candidate.py').write_text(candidate,newline='\n')
        spec.update(kind='extended-short-rows',short_limit=limit,origin_manifest=record(source/'manifest.json'),
                    numerical_note='Ascending-index FP32 FMA on short rows may differ from matrix-instruction reduction; no value or gate changes. Original numerical bounds apply.')
        write(folder/'spec.json',spec)
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)

if __name__=='__main__':main()
