"""Compile the retained revised recipe TeX and unchanged architecture panel."""
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=ROOT/'tmp/rewrite-recipe-build'
TEX=HERE/'figure-source'
OUT=HERE/'figures'


def main():
    for p in (WORK,TEX,OUT):p.mkdir(parents=True,exist_ok=True)
    for name in ('sparsification-ladder.tex','pythia-architecture-sparsification-ladder.tex','pythia-architecture-map.pdf'):
        shutil.copy2(TEX/name,WORK/name)
    for name in ('sparsification-ladder','pythia-architecture-sparsification-ladder'):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=WORK,capture_output=True,text=True)
        (WORK/(name+'-build.txt')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(f'{name}: see {WORK}')
        assert 'Overfull' not in result.stdout
        shutil.copy2(WORK/(name+'.pdf'),OUT/(name+'.pdf'))
    print('Recipe figure rebuilt; original architecture panel and all analytic values retained.')


if __name__=='__main__':
    main()
