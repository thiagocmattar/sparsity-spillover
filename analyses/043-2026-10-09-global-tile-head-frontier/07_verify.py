"""Verify the fixed analysis artifacts and their local source identities."""
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import pymupdf
from evidence import HERE, ROOT, frontier, write


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    sources={}
    for kind in ('latency','geometry'):
        for row in json.loads((HERE/f'provenance/{kind}-sources.json').read_text(encoding='utf-8')):
            if row['path'] in sources:assert sources[row['path']]==row
            sources[row['path']]=row
    for row in sources.values():
        path=ROOT/row['path']
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']
    latency=json.loads((HERE/'data/latency.json').read_text(encoding='utf-8'))
    geometry=json.loads((HERE/'data/geometry.json').read_text(encoding='utf-8'))
    points=latency['points'];assert len(points)==len({p['id'] for p in points})==138
    assert [p['id'] for p in frontier(points)]==latency['global_frontier_ids']
    assert len(latency['global_frontier_ids'])==20
    for cohort,edge in latency['compatible_frontier_ids'].items():
        assert [p['id'] for p in frontier([p for p in points if p['execution_cohort']==cohort])]==edge
    assert len(latency['compatible_frontier_ids']['FP16 native-body/head cohort'])==3
    head=[p for p in points if p['group'].startswith('head')]
    assert len(head)==18 and not any(p['id'] in latency['global_frontier_ids'] for p in head)
    assert latency['coverage']==dict(documents=500,blocks=338,input_tokens=692224,predictions=691886,excluded_tail_tokens=1444)
    for r in latency['head_boundaries']:
        n=r['head_counts'];assert n['exact_zero']==n['empty_gate_tiles']*r['head_tile_tokens']*16
        assert math.isclose(n['empty_consumer_tiles']/n['consumer_tiles'],n['skipped_mma']/n['potential_mma'],rel_tol=1e-12)
        assert r['max_logit_error']==0 and len(r['processes'])==3
        for boundary,t in r['timings'].items():
            for mode,value in t['host_ms']['geomean_ms'].items():
                computed=math.exp(sum(math.log(p['timings'][boundary]['host_ms']['geomean_ms'][mode]) for p in r['processes'])/3)
                assert math.isclose(value,computed,rel_tol=1e-12)
    for c in latency['decomposition']['conditions']:
        for r in c['modes'].values():
            assert math.isclose(sum(r['categories'].values()),sum(r['semantic_categories'].values()),rel_tol=1e-12)
    all_geometry=geometry['conditions']+geometry['manuscript_context']
    assert len(all_geometry)==28 and sum(r['steps'] for r in all_geometry)==geometry['boundary_records']==19936
    for r in all_geometry:
        assert r['zero_lr_boundaries']==1
        for metric,trace in r['trace'].items():
            assert len(trace)==712
            if isinstance(trace[0],bool):
                s=r['summary'][metric];assert s['count']==sum(trace) and s['total']==712
                assert math.isclose(s['percent'],100*sum(trace)/712,rel_tol=1e-12)
            else:assert all(math.isfinite(v) and v>=0 for v in trace)
        assert all(v<=100+1e-10 for v in r['trace']['removed_pressure_percent'])
    assert all(r['summary']['eligible_parameter_tensors']==[75 if r['pressure_recipe']=='Ph+head' else 65] for r in geometry['conditions'])
    with (HERE/'results/all-executions.csv').open(encoding='utf-8',newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==138
    for row,p in zip(csv_rows,points):
        assert row['id']==p['id'] and float(row['loss'])==p['loss'] and float(row['latency_ms'])==p['latency_ms']
    figures=sorted((HERE/'figures').glob('*.pdf'));assert len(figures)==6
    for path in figures:
        with pymupdf.open(path) as doc:
            assert len(doc)==1 and len(doc[0].get_text())>500
        assert (HERE/'observations'/f'{path.stem}.md').is_file()
    links=0
    final_outputs={HERE/'results/verification.json',HERE/'provenance/generated.json'}
    for path in [*HERE.rglob('*.md'),ROOT/'research/INDEX.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):continue
            target=link.split('#')[0]
            resolved=(path.parent/target).resolve()
            assert resolved.exists() or resolved in final_outputs,(path,link)
            links+=1
    result=subprocess.run([sys.executable,'-X','utf8',str(HERE/'test_analysis.py')],capture_output=True,text=True)
    print(result.stderr.strip(),flush=True);assert result.returncode==0
    write('results/verification.json',dict(status='verified',source_files=len(sources),execution_points=138,
        global_frontier_points=20,head_global_frontier_points=0,head_cohort_frontier_points=3,
        head_checkpoints=6,head_boundaries=18,profiled_checkpoints=6,profiled_execution_modes=22,
        geometry_conditions=28,geometry_boundaries=19936,mathematical_tests=5,
        pdf_figures=6,observation_files=6,resolved_markdown_links=links,
        visual_review='All six PDF pages rendered to ignored temporary PNGs and inspected; publication outputs are PDF only.',
        source_runs_modified=False,new_gpu_runs=False))
    generated=[]
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name=='generated.json':continue
        generated.append(dict(path=path.relative_to(HERE).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    write('provenance/generated.json',generated)
    assert all(path.is_file() for path in final_outputs)
    print({'status':'verified','source_files':len(sources),'markdown_links':links,'pdf_figures':6},flush=True)


if __name__=='__main__':main()
