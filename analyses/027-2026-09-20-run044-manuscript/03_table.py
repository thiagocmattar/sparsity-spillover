"""Complete the 14M-only appendix table while retaining every original value."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "data/full-trained-results.json").read_text(encoding="utf-8"))
    rows = [r for r in data["trained_points"] if r["model"] == "14M"]

    def group(scope, pressure):
        dose = "local_pressure_weight" if scope == "1" else "kappa"
        return sorted((r for r in rows if (r["scope"], r["pressure"]) == (scope, pressure)),
                      key=lambda r: r[dose])

    naive, ol1 = group("1", "L1"), group("1", "h")
    hz, four, seven = group("hz", "h"), group("4", "none"), group("7", "none")
    assert [r["local_pressure_weight"] for r in naive] == [r["local_pressure_weight"] for r in ol1] == [.05, .1, .5, 1]
    assert [r["kappa"] for r in hz] == [r["kappa"] for r in four] == [r["kappa"] for r in seven] == [0, .01, .05, .1, .5]
    lines = [
        "% Source: analyses/027-2026-09-20-run044-manuscript/observations/001-run044-integration.md; 03_table.py.",
        r"\begin{table}[!htbp]", r"\centering\small",
        r"\setlength{\tabcolsep}{6pt}", r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{\textbf{Additional 14M ablations.} Pressure at $h$ (top) and threshold sweeps (bottom). $T_2/P_h$ gates $h,z$ with OL1 at $h$ and $\lambda=1$; $T_4/P_0$ and $T_7/P_0$ use no pressure. Controls appear in Table~\ref{tab:endpoints-14m-70m}.}",
        r"\label{tab:endpoints-14m-only}", r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\multirow{2}{*}{$\lambda$} & \multicolumn{2}{c}{$T_1/P_1$ (L1)} & \multicolumn{2}{c}{$T_1/P_1$ (OL1)} & \multicolumn{2}{c}{} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) & \multicolumn{2}{c}{} \\",
        r"\midrule"]
    for a, b in zip(naive, ol1):
        lines.append(f"{a['local_pressure_weight']:g} & {a['loss']:.4f} & {a['sparsity']:.3f} & "
                     f"{b['loss']:.4f} & {b['sparsity']:.3f}" + r" & \multicolumn{2}{c}{} \\")
    lines += [
        r"\midrule",
        r"\multirow{2}{*}{$\kappa$} & \multicolumn{2}{c}{$T_2/P_h$} & \multicolumn{2}{c}{$T_4/P_0$} & \multicolumn{2}{c}{$T_7/P_0$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r" & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) \\",
        r"\midrule"]
    for a, b, c in zip(hz, four, seven):
        lines.append(f"{a['kappa']:g} & {a['loss']:.4f} & {a['sparsity']:.3f} & "
                     f"{b['loss']:.4f} & {b['sparsity']:.3f} & {c['loss']:.4f} & {c['sparsity']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    target = HERE / "tables/14m-only.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("Generated 23 additional 14M endpoints; other complete-results tables remain unchanged")


if __name__ == "__main__":
    main()
