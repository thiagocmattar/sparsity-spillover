# Pythia 14M, 31M and 70M: loss versus full-model latency

User-requested extension of Analysis 034 Figure 02 with the six Run 054
checkpoints (30,494,720 parameters): Base and T2/Ph at kappa 0,0.01,0.05,0.1,0.5.
Base appears under both kernel and PyTorch execution. Historical coordinates
are preserved exactly, yielding 75 execution points for 72 checkpoints.

Complete: all six training endpoints and 18 timing processes are verified.
No preflight result is plotted. The manuscript is unchanged.

- [Updated publication PDF](figures/01-14m-31m-70m-latency-quality.pdf)
- [Question, method, caption, results and caveats](observations/001-combined-scales.md)
- [Exact new endpoints and evidence hashes](data/31m-results.json)
- [Combined coordinates and measured Pareto frontier](data/combined-figure.json)

Quality is canonical final-checkpoint FP16 validation over all 338 complete
2048-token blocks from 500 MiniPile validation documents; 1444 tail tokens are
excluded. Timings use RTX 5090, BF16, B1/T2048 and full logits, 64 inputs x seven
passes x three fresh processes. The new kernel specializes Run 045 opt073 for
width 256, FFN 1024 and head dimension 32. All processes must pass full-block
qualification against native eager. Training runs concurrently on six H200s.

`01_collect.py` verifies the new cohort, checkpoint identity, raw timing reduction,
process coverage and numerical qualification. `02_plot.py` reproduces the
original figure's recipe colors, logarithmic axes and old coordinates, adding
triangular markers for 31M. Per-process latency dispersion remains in the data.
No timing rescaling, cross-size curve fit or extrapolation is introduced.

The new scale contributes five nominal Pareto points: Base PyTorch and T2/Ph
at kappa 0,0.01,0.05,0.1. Base PyTorch is 1.023861 ms at loss 4.565514; kappa=0.1
is 0.908418 ms at loss 4.757208. The high-threshold kappa=0.5 endpoint is dominated
by existing 14M points. This fills the measured loss-latency gap between sizes,
but different sessions and kernels, one seed, and only three sizes limit any
scaling-law interpretation. Tiny latency differences near Base are not a
statistical claim of superiority.

Reproduce from the repository root after restoring the retained checkpoints:

```powershell
.venv/Scripts/python.exe runs/054-2026-09-24-pythia31m-t2-ph/prelaunch/verify-cohort.py
.venv/Scripts/python.exe analyses/037-2026-09-24-14m-31m-70m-latency-quality/01_collect.py
.venv/Scripts/python.exe analyses/037-2026-09-24-14m-31m-70m-latency-quality/02_plot.py
```

The output is PDF only. The PDF was rendered with Poppler and visually checked;
temporary raster QA is under `tmp/pdfs/run054/`. All 75 execution coordinates
are within the axes, and the original Analysis034 PDF hash remains unchanged.
