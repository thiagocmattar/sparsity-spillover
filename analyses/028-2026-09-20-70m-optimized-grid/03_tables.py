"""Compact endpoint tables with explicit implementation-specific 70M timings."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EVIDENCE='% Source: Analysis028 observations/001-matched-70m-grid.md; 03_tables.py; data/full-trained-results.json.'


def write(name,lines):
    target=HERE/'tables'/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join([EVIDENCE]+lines+['']),encoding='utf-8',newline='\n')


def recipe(scope,pressure):
    s='2' if scope=='hz' else scope
    p={'none':'0','h':'h','all':r'\mathrm{all}'}[pressure]
    return rf'$T_{{{s}}}/P_{{{p}}}$'


def latency(row,mode):
    return f"{row['implementation_latency_ms'][mode]:.3f}" if row['qualified'][mode] else r'$\dagger$'


def main():
    data=json.loads((HERE/'data/full-trained-results.json').read_text())
    rows=data['trained_points']
    index={(r['model'],r['scope'],r['pressure'],r['kappa']):r for r in rows if not r.get('appendix_only')}
    lines=[r'\begin{table}[H]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
           r'\renewcommand{\arraystretch}{1.02}',
           r'\caption{\textbf{Matched 14M/70M results.} OL1 pressure uses $\lambda=b=1$. The 70M port and optimized latencies use the same RTX\,5090 session; the native base takes '+
           f"{data['matched_summary']['native_base_ms']:.3f}"+r'\,ms. The additional 70M $T_2/P_h$, $\kappa=0.5$ endpoint is reported separately in Table~\ref{tab:70m-additional-endpoint}; its optimized latency is unmeasured.}',
           r'\label{tab:endpoints-14m-70m}',r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',
           r'\multirow{2}{*}{Recipe} & \multirow{2}{*}{$\kappa$} & \multicolumn{2}{c}{14M} & \multicolumn{2}{c}{70M} & \multicolumn{2}{c}{70M latency (ms)} \\',
           r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}',
           r' & & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) & Port & Optimized \\',r'\midrule']
    groups=[('0','none'),('1','none'),('hz','h'),('4','h'),('4','all'),('7','h'),('7','all')]
    for number,(scope,pressure) in enumerate(groups):
        doses=[None] if scope in ('0','1') else [0,.01,.05,.1,.5]
        for i,dose in enumerate(doses):
            a=index[('14M',scope,pressure,dose)];b=index.get(('70M',scope,pressure,dose))
            label=recipe(scope,pressure)
            if len(doses)>1:label=rf'\multirow{{5}}{{*}}{{{label}}}' if i==0 else ''
            elif scope=='0':label+=' (Base)'
            else:label+=' (ReLU)'
            values=f"{a['loss']:.4f} & {a['sparsity']:.3f}"
            if b:
                values+=f" & {b['loss']:.4f} & {b['sparsity']:.3f} & {latency(b,'legacy_graph')} & {latency(b,'candidate_graph')}"
            else:values+=' & --- & --- & --- & ---'
            dose_text='---' if dose is None else f'{dose:g}'
            lines.append(f'{label} & {dose_text} & {values}'+r' \\')
        if number<len(groups)-1:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write('14m-70m.tex',lines)

    # T2 now belongs in the matched-size table, eliminating duplicated rows.
    lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{6pt}',
           r'\renewcommand{\arraystretch}{1.05}',
           r'\caption{\textbf{Additional 14M ablations.} Pressure at $h$ (top) and thresholding without pressure (bottom). Controls and $T_2/P_h$ appear in Table~\ref{tab:endpoints-14m-70m}.}',
           r'\label{tab:endpoints-14m-only}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
           r'\multirow{2}{*}{$\lambda$} & \multicolumn{2}{c}{$T_1/P_1$ (L1)} & \multicolumn{2}{c}{$T_1/P_1$ (OL1)} \\',
           r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
           r' & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) \\',r'\midrule']
    for dose in (.05,.1,.5,1):
        pair=[next(r for r in rows if r['model']=='14M' and r['scope']=='1' and
                   r['pressure']==p and r['local_pressure_weight']==dose) for p in ('L1','h')]
        lines.append(f'{dose:g} & '+' & '.join(f"{r['loss']:.4f} & {r['sparsity']:.3f}" for r in pair)+r' \\')
    lines += [r'\midrule',r'\multirow{2}{*}{$\kappa$} & \multicolumn{2}{c}{$T_4/P_0$} & \multicolumn{2}{c}{$T_7/P_0$} \\',
              r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',r' & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) \\',r'\midrule']
    for dose in (0,.01,.05,.1,.5):
        pair=[index[('14M',scope,'none',dose)] for scope in ('4','7')]
        lines.append(f'{dose:g} & '+' & '.join(f"{r['loss']:.4f} & {r['sparsity']:.3f}" for r in pair)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write('14m-only.tex',lines)

    lines=[r'\begin{table}[!htbp]',r'\centering\small',
           r'\caption{\textbf{70M implementation comparison.} Representative rows from the complete 26-checkpoint sweep in Table~\ref{tab:endpoints-14m-70m}. All timings are full-model geometric means from the same RTX\,5090 session. Each implementation preserves the checkpoint and thresholds.}',
           r'\label{tab:70m-kernel-transfer}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
           r'Checkpoint & Native (ms) & Port (ms) & Optimized (ms) & Native-base speedup \\',r'\midrule']
    for cid in ('c00','c25','c16','c21'):
        r=next(r for r in rows if r.get('run045_id')==cid)
        label=recipe(r['scope'],r['pressure'])+(rf", $\kappa={r['kappa']:g}$" if r['kappa'] is not None else '')
        t=r['implementation_latency_ms']
        speed=r['native_base_speedup']['candidate_graph']
        speed_text=rf'${speed:.3f}\times$' if speed is not None else r'$\dagger$'
        lines.append(f"{label} & {latency(r,'native_graph')} & {latency(r,'legacy_graph')} & {latency(r,'candidate_graph')} & {speed_text}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write('70m-kernel-transfer.tex',lines)

    r, = [r for r in rows if r.get('appendix_only')]
    write('70m-additional-endpoint.tex',[
        r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Additional 70M endpoint.} $T_2/P_h$ at $\kappa=0.5$, outside the matched optimized-kernel sweep. Original-port and same-checkpoint native timings share a separate RTX\,5090 session (Run046), with three qualified processes and 1,344 timings per implementation. Optimized latency was not measured.}',
        r'\label{tab:70m-additional-endpoint}',
        r'\begin{tabular}{@{}rrrr@{}}',r'\toprule',
        r'Loss & $\Smodel$ (\%) & Port (ms) & Native, same checkpoint (ms) \\',r'\midrule',
        f"{r['loss']:.4f} & {r['sparsity']:.3f} & {r['original_port_latency_ms']:.3f} & {r['native_same_checkpoint_ms']:.3f}"+r' \\',
        r'\bottomrule',r'\end{tabular}',r'\end{table}'])


if __name__=='__main__':main()
