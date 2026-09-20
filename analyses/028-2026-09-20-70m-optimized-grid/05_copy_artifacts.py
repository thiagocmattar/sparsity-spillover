"""Install the three approved figures and their data/tables with source hashes."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DRAFT=ROOT/'manuscript/draft'


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    return dict(source=source.relative_to(ROOT).as_posix(),sha256=digest,bytes=source.stat().st_size)


def main():
    for folder,names,script in [
        ('figures',{'23-70m-quality-sparsity-native-latency.pdf':'23-70m-quality-sparsity-native-latency.pdf',
                    '03-pressure-scope-threshold.pdf':'03-pressure-scope-threshold.pdf',
                    '20-kernel-structure-native-base-speedup.pdf':'20-kernel-structure-native-base-speedup.pdf'},'02_figures.py'),
        ('supplementary-data',{'all-model-quality-sparsity.json':'full-trained-results.json',
                              '70m-quality-sparsity-native-latency.json':'70m-quality-sparsity-native-latency.json',
                              'paired-pressure-figure-data.json':'paired-pressure-figure-data.json',
                              'kernel/70m-retained-controls.json':'retained-controls.json'},'01_collect.py')]:
        sources=DRAFT/folder/'SOURCES.json'
        records=json.loads(sources.read_text())
        for name,original in names.items():
            record=copy(HERE/('figures' if folder=='figures' else 'data')/original,DRAFT/folder/name)
            observation='002-retained-14m-diagnostics.md' if name.startswith('20-') else '001-matched-70m-grid.md'
            record.update(observation=(HERE/'observations'/observation).relative_to(ROOT).as_posix(),
                          script=(HERE/('04_controls.py' if original=='retained-controls.json' else script)).relative_to(ROOT).as_posix())
            records[name]=record
        sources.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name in ('14m-only.tex','14m-70m.tex','70m-kernel-transfer.tex','70m-additional-endpoint.tex'):
        copy(HERE/'tables'/name,DRAFT/'tables/compact-results'/name)
    print('Copied three figures, four data exports and four compact tables; hashes match.')


if __name__=='__main__':main()
