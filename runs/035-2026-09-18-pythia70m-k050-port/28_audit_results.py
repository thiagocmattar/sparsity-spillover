"""Summarize complete validation and diagnostic coverage after verified retrieval."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    verification=json.loads((HERE/'artifacts/verification.json').read_text())
    assert verification['all_required_artifacts_verified'] and verification['qualified_processes']==66
    candidate=[];native=[];deltas=[];sources=[];diagnostics=[]
    for p in sorted((HERE/'artifacts/attempts').glob('scientific-*/quality.json')):
        q=json.loads(p.read_text());assert all(q['pass'].values())
        assert all(len(v)==338 for v in q['gates'].values())
        candidate+=q['gates']['candidate_graph'];native+=q['gates']['native_graph']
        deltas.append(abs(q['loss_delta']['candidate_graph']))
        sources.append({'path':p.relative_to(HERE).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for p in sorted((HERE/'artifacts/attempts').glob('scientific-*-r1-001/diagnostics.json')):
        d=json.loads(p.read_text());assert d['status']=='complete'
        assert d['coverage']=={'blocks':338,'documents':500,'excluded_tail_tokens':1444,'input_tokens':692224}
        assert d['native_weight_statistics'] and d['pooled_by_site'] and d['per_site_layer']
        diagnostics.append(p.parent.name)
        sources.append({'path':p.relative_to(HERE).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert len(deltas)==66 and len(diagnostics)==22 and len(candidate)==len(native)==22308
    assert all(b['pass'] and b['finite'] and b['elementwise_gate'] for b in candidate+native)
    report={'qualified_processes':66,'qualified_checkpoints':22,'validation_blocks_per_process':338,
        'candidate_blocks_checked':len(candidate),'candidate_bit_exact_blocks':sum(b['max_abs']==0 for b in candidate),
        'native_graph_bit_exact_blocks':sum(b['max_abs']==0 for b in native),
        'candidate_max_absolute_logit_error':max(b['max_abs'] for b in candidate),
        'candidate_max_relative_l2':max(b['relative_l2'] for b in candidate),
        'candidate_max_absolute_loss_delta':max(deltas),
        'elementwise_rule':'abs(candidate-reference)<=0.25+0.02*abs(reference), plus relative L2<=0.02; absolute loss delta<=0.001',
        'diagnostic_checkpoints':diagnostics,'sources':sources}
    (HERE/'results/qualification-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'sources','diagnostic_checkpoints'}}))


if __name__=='__main__':main()
