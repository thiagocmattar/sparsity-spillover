"""Rebuild the nine task.md PDFs from local, checkpoint-joined evidence only.

Usage (repository root): .venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/13_rebuild_paper_figures.py
"""
import hashlib
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import paper_evidence
import paper_style
import paper_quality_figures
import paper_effect_figures
import paper_execution_figures

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def architecture(precompiled=None):
    scratch=ROOT/'tmp/pdfs/analysis024-architecture'
    scratch.mkdir(parents=True,exist_ok=True)
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={scratch}',str(HERE/'paper-architecture.tex')]
    if precompiled is None:
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout+'\n'+completed.stderr)
    destination=HERE/'figures/02-intervention-sites.pdf'
    shutil.copyfile(precompiled or scratch/'paper-architecture.pdf',destination)
    return {'file':destination.name,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'source':'paper-architecture.tex','adapted_from':'manuscript/draft/figures/pythia-architecture-map.tex'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--architecture-pdf',type=Path,help='PDF just compiled from paper-architecture.tex, when TeX requires a separate approved invocation.')
    args=parser.parse_args()
    data=paper_evidence.build()
    paper_style.setup()
    figures=[paper_quality_figures.make(data),architecture(args.architecture_pdf),
             paper_effect_figures.pressure_scope(data),paper_effect_figures.operation_changes(data),
             paper_execution_figures.quality_latency(data),paper_quality_figures.make(data,full=True),
             paper_effect_figures.pressure_none(data),paper_execution_figures.speedup(data),
             paper_execution_figures.instruction(data)]
    result={'figures':figures,'table2':paper_effect_figures.table2(data),
            'comparison_checkpoint_count':sum(r['comparison_cohort'] for r in data['checkpoints']),
            'checkpoint_table_sha256':hashlib.sha256((HERE/'data/paper-checkpoints.json').read_bytes()).hexdigest(),
            'source_script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(__file__),*sorted(HERE.glob('paper_*.py')),HERE/'paper-architecture.tex']}}
    (HERE/'data/paper-derived.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Generated nine PDFs and Table 2; saved exact pairs, counters, selections, and hashes.')


if __name__=='__main__':
    main()
