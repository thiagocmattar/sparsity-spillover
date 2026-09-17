"""Small TeX evidence tables for explicit threshold/pressure scope."""
import json
from evidence import HERE, KAPPAS, FAMILIES


def recipe(f):
    if f.startswith(("A4", "A7")):
        t=f[1];p="h" if f.endswith("-H") else t if "OL1" in f else "0"
        return rf"$T_{t}/P_{p}$"
    return f


def write_tables(data):
    out=HERE/"tables";out.mkdir(exist_ok=True)
    lines=[r"% Source: Analysis 021 pressure-scope/tables.py; Analysis 023 approved losses.",
           r"\begin{tabular}{@{}rrrrr@{}}",r"\toprule",
           r" & \multicolumn{2}{c}{$P_0$: no pressure} & \multicolumn{2}{c}{$P_h$: h-only OL1} \\",
           r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
           r"$\kappa$ & $\Delta L$ & $\Delta\Smodel$ (pp) & $\Delta L$ & $\Delta\Smodel$ (pp) \\",r"\midrule"]
    for k in KAPPAS:
        selected=[next(r for r in data['contrasts'] if r['kappa']==k and r['treatment']==a and r['reference']==b)
                  for a,b in [('A7','A4'),('A7-OL1-H','A4-OL1-H')]]
        cells=[f"{k:g}"]+[v for r in selected for v in (f"{r['delta_loss']:+.4f}",f"{r['delta_s_pp']:+.3f}")]
        if k==.5:cells=[r"\textbf{"+c+"}" for c in cells]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r"\bottomrule",r"\end{tabular}"]
    (out/'threshold-scope-contrasts.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8', newline='\n')

    endpoints=[{**r,"kappa":r['dose'],"loss":r['loss'],"S_model_percent":100*r['R_model']} for r in data['original_trained']]
    endpoints += [{**r,"scale":"14M","dose":r['kappa']} for r in data['rows'] if r['family'].endswith('-H')]
    order=['A0','A1-H','A1-H-L1','A1-H-OL1',*FAMILIES]
    endpoints.sort(key=lambda r: (['14M','70M','410M'].index(r['scale']),order.index(r['family']),r['dose'] or 0))
    assert len(endpoints)==64
    caption=(r"All 64 trained endpoints. $T_i/P_j$ specifies threshold and pressure scope; $P_0$ means no pressure. "
             r"$U_{\mathrm{arch}}$ uses the common seven-site ceiling within each size. "
             r"H-only losses use ordinary final validation; other losses use the logical diagnostic pass "
             r"(Appendix~\ref{app:evaluation}). Rounded zero sparsity need not be exactly zero.")
    head=[r"\toprule",r"Size & Recipe & Parameter & Loss & $\Smodel$ (\%) & $U_{\mathrm{arch}}$ (\%) \\",r"\midrule"]
    lines=[r"% Source: Analysis 021 pressure-scope/tables.py; 54 original + 10 verified h-only endpoints.",
           r"{\small\renewcommand{\arraystretch}{0.95}",r"\begin{longtable}{@{}lllrrr@{}}",
           r"\caption{"+caption+r"}\label{tab:trained-endpoints}\\",*head,r"\endfirsthead",
           r"\multicolumn{6}{l}{\tablename~\thetable{} (continued)}\\",*head,r"\endhead",
           r"\bottomrule\endfoot"]
    for r in endpoints:
        dose=r['dose'];p='---' if dose is None else rf"$\{'lambda' if r['family'].startswith('A1') else 'kappa'}={dose:g}$"
        ceil=next(c['R_model_max_fraction'] for c in data['ceilings'] if c['scale']==r['scale'] and c['family']=='A7')
        lines.append(f"{r['scale']} & {recipe(r['family'])} & {p} & {r['loss']:.4f} & {r['S_model_percent']:.3f} & {r['S_model_percent']/ceil:.3f}"+r" \\")
    lines += [r"\end{longtable}}"]
    (out/'trained-endpoints.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8', newline='\n')

    lines=[r"% Source: Analysis 021 pressure-scope/tables.py; matched reported losses.",
           r"\begin{table}[p]\centering\small",
           r"\caption{Pressure-scope increments and threshold-placement contrasts at 14M. Each row subtracts the reference at the same $\kappa$. These are single-seed comparisons.}\label{tab:pressure-scope-full}",
           r"\begin{tabular}{@{}llrr@{}}",r"\toprule",r"Treatment $-$ reference & $\kappa$ & $\Delta L$ & $\Delta\Smodel$ (pp) \\",r"\midrule"]
    for r in data['contrasts']:
        if (r['treatment'],r['reference'])==('A7','A4'):continue
        lines.append(f"{recipe(r['treatment'])} $-$ {recipe(r['reference'])} & {r['kappa']:g} & {r['delta_loss']:+.4f} & {r['delta_s_pp']:+.3f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (out/'pressure-scope-contrasts.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8', newline='\n')
    lines=[r"% Source: Analysis 021 pressure-scope/evidence.py; Run 029 raw timing pairs.",
           r"\begin{table}[htbp]\centering\small",
           r"\caption{Restored four-site h-only runtime measurements. All entries use full-model geometric-mean latency in ms, pooled from the original matched timing pairs. These five records extend the original 30-checkpoint analysis without new execution.}\label{tab:h-only-runtime}",
           r"\begin{tabular}{@{}rrrrrr@{}}",r"\toprule",r"$\kappa$ & Loss & Native & All skips off & Projection only & Full K050 \\",r"\midrule"]
    for r in data['h_only_runtime']:
        lines.append(f"{r['kappa']:g} & {r['validation_loss']:.4f} & {r['native_baseline_gm_ms']:.5f} & {r['all_skips_off_candidate_gm_ms']:.5f} & {r['projection_on_candidate_gm_ms']:.5f} & {r['full_candidate_gm_ms']:.5f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (out/'h-only-runtime.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8', newline='\n')


if __name__=='__main__':
    write_tables(json.loads((HERE/'data/evidence.json').read_text()))
