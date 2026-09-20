# Analysis026: 14M quality, sparsity and latency with h/z thresholds

User-requested extension of manuscript Figure06's two-panel layout with every
14M trained model in panel (a) of Figure14 and the four completed Run041 models.
The user explicitly chose **40 trained models total**, retaining Figure14's
exclusion of the four historical naive-L1 models.

- [Publication PDF](figures/01-14m-quality-sparsity-latency-with-hz.pdf)
- [Caption, results and limitations](observations/001-quality-sparsity-latency.md)
- [Exact plotted data and SHA-256 provenance](data/figure-data.json)
- [Generating script](01_build.py)

Both panels contain the same 40 checkpoints: 36 existing models plus four
h/z-gated, h-only OL1 models at kappa=0/.01/.05/.1 and lambda=1. The quality
panel also retains the two existing ten-point Base/ReLU clipping trajectories;
eight high-loss clipping points continue above its view, as in the reference.
The latency panel shows trained models, matching Figure06's layout. Run036's
existing clipping timings are outside this figure's trained-model latency view.
Run041 has no post-hoc clipping.

All trained points are visible. Historical colors and line styles are retained;
the new dark-blue `T_hz/P_h` series denotes one-sided h/z gates and OL1 at h.
The figure adds its analytic 5.3471% reach guide alongside the T4/T7 ceilings.
Model-wide sparsity is count-pooled `100 * R_model`, including the dense LM-head
denominator. It is distinct from activation zero fractions and measured speedup.

Losses use ordinary reloaded final-checkpoint validation. Full validation covers
338 complete blocks from 500 documents, with 1,444 tail tokens excluded. K050
latencies use geometric means of 1,344 host timings per checkpoint on RTX5090,
BF16, B=1, T=2048, full logits. Runs029/033/041 contribute 31/5/4 models from
different physical GPU/host sessions; absolute latency is descriptive across
sessions. The common seed, initialization, data order and training budget are
verified, but there is no training-seed uncertainty estimate.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/026-2026-09-20-14m-hz-quality-sparsity-latency/01_build.py
```

The builder checks the exact cohort, checkpoint identities, final losses,
integer product counters, coverage, shared initialization/data schedule, every
Run041 qualified process and its raw timing reduction, and all dose grids. It
records 149 source-file hashes and checks that both reference PDFs are unchanged.
The final PDF is rendered and visually inspected; fonts and text bounds are
checked separately in `data/verification.json`.

Recheck the data identities and PDF structure with the installed system Python
(which provides PyMuPDF):

```powershell
python -X utf8 analyses/026-2026-09-20-14m-hz-quality-sparsity-latency/02_verify.py
```

This is local reprocessing of retained evidence. No training, evaluation,
latency benchmark, cloud work, manuscript edit or finding promotion is performed.
