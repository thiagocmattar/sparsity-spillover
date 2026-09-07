# Analysis 017: Pythia-14M auto-research history, K001-K050

Status: completed descriptive replot of existing evidence; no new experiment
or GPU launch, and no manuscript/finding promotion.

## Requested question and scope

The user requested a single progress plot like Run028 Figure06, but with
the kernel-iteration axis beginning at 1 and including earlier runs. The
existing Pythia-14M scope is retained. This analysis reconstructs the
available 14M full-model timings from Runs025-028, without mixing in 70M,
410M, isolated primitives, or eager-to-graph cross-mode speedup ratios.

The common formula is **native latency / candidate latency**, paired on
the same input and matched execution mode within each phase. Hardware and
execution change across the historical sequence, so three clearly labeled
phases are necessary in the single axes:

| Candidate IDs | Hardware | Execution | Points |
| --- | --- | --- | ---: |
| K001-K016 | RTX PRO 4500 Blackwell | Eager / eager | 56 |
| K017-K030 | RTX5090 | Eager / eager | 61 |
| K031-K050 | RTX5090 | CUDA graph / CUDA graph | 136 |

This is not one fixed-baseline experiment. The figure keeps separate
best-so-far traces at hardware/execution boundaries; it does not connect
them into an apparent gain or regression caused by kernel changes alone.
The graph-only comparable view remains
[Run028 Figure07](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/figures/07-autoresearch-kernel-progress-grid.pdf).

## Files and reproduction

- `01_reduce.py`: source verification, raw paired timing reduction, retry
  selection, fixed-checkpoint identity check, and phase-local incumbents.
- `02_plot.py`: one scatter plot with a light grid, no bars/boxes/quantiles
  or variant legend, and visible protocol boundaries.
- `results/progress.json`: all 253 plotted points, statuses, coverage,
  source hashes, early attempt audit, superseded retries, and missing IDs.
- `results/figure-provenance.json`: source/script/PDF hashes and presentation.
- `figures/01-kernel-progress-all-iterations.pdf`: vector publication output.
- [Observation01](observations/01-kernel-progress-all-iterations.md): complete
  source selection, caption, numerical results, and caveats.
- `test_progress.py`: pairing, retry selection, incumbent boundaries,
  missing-data semantics, and exact point/layout tests.

From the repository root:

```powershell
.venv\Scripts\python.exe analyses/017-2026-09-07-kernel-progress-all-iterations/01_reduce.py
.venv\Scripts\python.exe analyses/017-2026-09-07-kernel-progress-all-iterations/02_plot.py
```

## Result and coverage limits

The figure has 253 model/setting measurements across 39 candidate IDs:
193 pass their recorded numerical check and 60 fail it. Of all points,
230 have full 338-block coverage; 23 have only their historical 8- or
16-block development check. Passing a partial check is not full validation.
Only passing full-validation c30 results can update the incumbent line.

K002-K005, K007-K010, and K014-K016 have no plotted 14M full-model timing.
Those IDs include other-size work and primitive/failed diagnostics in the
earlier multi-model search. They retain their ordinal positions but no
fabricated speedup. The figure spans K001-K050 without claiming 50 numeric
14M measurements. Multiple points at K011 include distinct tested masks;
they are not independent checkpoint/training replicates.

Within the matched-graph phase, the fixed-checkpoint incumbent improves
from 1.347607x to 1.738990x. The eager phase's 2.274083x incumbent must not
be directly ranked against 1.738990x over the faster graph baseline.
All unfavorable measured points remain visible under the declared
coverage-based retry and final-cohort precedence.

The PDF skill's complete render-and-inspect workflow was used, including
adjusting title/subtitle spacing. Both this figure and the Run028 grid view
retain their original numerical evidence. This is an adaptive engineering
history, not evidence of causal agent advantage, a universal sparsity speed
law, or a controlled comparison across hardware. A homogeneous curve from
K001 would require newly timing earlier compatible candidates under a
single agreed hardware/reference/validation protocol.

Verification: five focused analysis tests and the Run028 grid-layout test
pass. The complete bootstrap + Run028 + Analysis017 suite passes 308 tests
in 9.70s. Final PDFs are one page each with embedded TrueType fonts; their
source/script/PDF hashes are verified before handoff. Prior plots and frozen
scientific evidence remain unchanged.
