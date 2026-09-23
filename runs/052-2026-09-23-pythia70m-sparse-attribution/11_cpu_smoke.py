"""Actual retained checkpoints and backbone installers; no CUDA execution."""
import argparse,gc
from support import RUN,BASE,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=('c00','c24','c25','d05','d10'));a=p.parse_args()
    import bootstrap,torch
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata
    torch.set_num_threads(4)
    row=bootstrap.checkpoint(a.condition);root=RUN if a.condition.startswith('d') else BASE
    for record in row['files']+row['provenance']:bootstrap.base_io.verify(record,root)
    with torch.inference_mode():
        model=load_checkpoint_pythia(AutoModelForCausalLM,root/row['checkpoint'],torch=torch).eval()
        topology=topology_metadata(model)
        actual=model(input_ids=torch.tensor([[0,1,2,3]]),use_cache=False).logits
        assert actual.shape==(1,4,50304) and bool(torch.isfinite(actual).all())
        model.to(dtype=torch.bfloat16);bootstrap.install_backbone(model)
        gates=[{'h':l._run026_joint.gh,'z':l._run026_joint.gz,'th':l._run026_joint.th,'tz':l._run026_joint.tz} for l in model.gpt_neox.layers]
        if a.condition=='c00':assert all(not g['h'] and not g['z'] for g in gates)
        else:
            threshold=.05 if a.condition in ('c24','d05') else .1
            assert all(g['h'] and g['z'] and g['th']==threshold and g['tz']==threshold for g in gates)
        write(RUN/'prelaunch'/f'cpu-smoke-{a.condition}.json',{'status':'passed','condition':a.condition,'topology':topology,'gates':gates,
              'scope':'Actual CPU native checkpoint forward and backbone installation. No new CUDA operator or latency qualification.'})
        print('CPU checkpoint and installation passed:',a.condition)


if __name__=='__main__':main()
