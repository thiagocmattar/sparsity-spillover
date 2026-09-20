"""GPU compatibility/skip-ablation checks, including original-kernel parity."""
import itertools
import torch
import adapter
from test_adapter import joint_reference
from run027_common import RUN,write_json


def main():
    torch.manual_seed(2702)
    rows=[]
    with torch.inference_mode():
        for gh,gz,skip in itertools.product([False,True],repeat=3):
            for pattern in ['random','zeros','equal','negative']:
                h=torch.randn(33,512,device='cuda',dtype=torch.bfloat16)
                z=torch.randn(33,128,device='cuda',dtype=torch.bfloat16)
                if pattern=='zeros': h.zero_(); z.zero_()
                if pattern=='equal': h.fill_(.5); z.fill_(.5)
                if pattern=='negative': h.fill_(-.5); z.fill_(-.5)
                wh=torch.randn(512,128,device='cuda',dtype=torch.bfloat16)*.02
                wz=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.02
                bh=torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
                bz=torch.randn_like(bh); r=torch.randn(33,128,device='cuda',dtype=torch.bfloat16)
                out=torch.empty_like(r); expected=torch.empty_like(r)
                args=(h,z,wh,wz,bh,bz,r,out,.5,.5,gh,gz,skip)
                adapter.extension().forward(*args)
                joint_reference(*args[:7],expected,*args[8:])
                error=float((out.float()-expected.float()).abs().max())
                passed=torch.allclose(out,expected,rtol=.02,atol=.125)
                if gh and gz and skip:
                    original=torch.empty_like(out)
                    adapter.joint.extension().forward(*args[:7],original,.5,.5)
                    passed=passed and torch.equal(out,original)
                rows.append({'gate_h':gh,'gate_z':gz,'skip':skip,'pattern':pattern,'max_abs':error,'pass':bool(passed)})
    write_json(RUN/'runtime/primitives.json',{'cases':rows,'pass':all(r['pass'] for r in rows)})
    if not all(r['pass'] for r in rows):raise RuntimeError('Primitive gate failed')
    print(f'{len(rows)} primitive cases passed',flush=True)


if __name__=='__main__':main()
