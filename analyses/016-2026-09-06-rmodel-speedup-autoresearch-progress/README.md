# Analysis 016: `R_model`, speedup, and autoresearch progress

Status: **complete descriptive synthesis; not promoted to a finding or a
result-bearing manuscript claim**.

## Question

This focused paper-facing synthesis asks two separate questions from the
completed Run 025 evidence:

1. how does canonical `R_model` associate with measured batch-one full-model
   speedup across Pythia-14M, 70M, and 410M on RTX PRO 4500 and H100 NVL; and
2. how did the correctness-constrained autoresearch trajectory change a common
   sparse-endpoint speed score across candidate milestones?

`R_model` remains the count-pooled fraction of logical multiplication
opportunities with an exact-zero operand. Speedup remains native eager latency
divided by sparse-candidate latency. Neither quantity is substituted for the
other.

## Sources and reduction

- Source run: [Run 025](../../runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/README.md).
- Detailed source analysis: [Analysis 015](../015-2026-09-06-pythia-agentic-kernel-search/README.md).
- [`01_reduce.py`](01_reduce.py) rechecks the integer `R_model` ratios, copies
  the 54 fresh-process checkpoint summaries and six primary qualified fits,
  and reconstructs the candidate trajectory from immutable Run 025 artifacts.
- [`source-provenance.json`](source-provenance.json) records source paths,
  sizes, and SHA-256 identities.
- [`02_plot.py`](02_plot.py) creates both PDFs using the established
  Analysis-010/015 typography and model-size palette.

The progress score is deliberately narrow and comparable: the geometric mean
of A4-OL1 `kappa=0.5` and A7-OL1 `kappa=0.5` batch-one, sequence-2,048,
full-model speedups on the RTX PRO 4500. Candidate IDs without both
correctness-qualified endpoint timings have no numeric score. The plotted
timings are the measurements available at each adaptive search milestone and
use 48--160 paired samples per endpoint; this is not a controlled comparison
of optimizers or search algorithms.

## Figures

- [Figure 1: `R_model` versus full-model speedup](figures/01-rmodel-vs-speedup.pdf)
- [Figure 2: autoresearch speedup progress](figures/02-autoresearch-speedup-progress.pdf)

Matching figure records are in [observations](observations/INDEX.md).

## Results

The fresh-process fits reproduce the Analysis 015 result while making the
cross-size/hardware contrast explicit:

| GPU | Size | Qualified points | Slope per +10 pp `R_model` | OLS `R2` | Best qualified speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX PRO 4500 | 14M | 11 | +0.0347x | 0.347 | 1.0544x |
| H100 NVL | 14M | 6 | +0.0649x | 0.321 | 1.0725x |
| RTX PRO 4500 | 70M | 9 | +0.0036x | 0.0027 | 1.0331x |
| H100 NVL | 70M | 6 | +0.0100x | 0.051 | 1.0452x |
| RTX PRO 4500 | 410M | 12 | +0.0105x | 0.636 | 1.0192x |
| H100 NVL | 410M | 6 | +0.0075x | 0.700 | 0.9326x |

The common sparse-endpoint trajectory is also heterogeneous:

| Size | First valid paired milestone | Best qualified milestone | Frozen final |
| --- | ---: | ---: | ---: |
| 14M | P0, 1.0927x | K001, 1.1382x | K013, 1.0461x |
| 70M | K004, 0.9446x | K016, 1.0213x | K016, 1.0213x |
| 410M | K004, 0.5856x | K010, 1.0147x | K010, 1.0147x |

Thus the loop finds useful specializations for 70M and 410M, but it does not
improve monotonically. At 14M the fastest common-endpoint candidate is K001;
the final K013 policy trades away speed to satisfy the broader correctness
contract. K012 makes the risk visible: its two plotted high-sparsity endpoints
pass, while three of the six complete-validation development conditions fail.

## Interpretation boundary

Figure 1 is descriptive within one training seed per checkpoint. Fresh Python
processes measure systems variability, not training uncertainty. H100 has six
endpoint sentinels per model size while RTX has all twelve checkpoints.

Figure 2 is one adaptive, architecture-specific engineering trajectory. It
shows both gains and regressions, but it cannot estimate the causal advantage
of agentic search relative to a human tuner or another optimizer. QK/PV
attention stayed dense, and the strongest compiled-dense comparator remains
outside the achieved Run 025 scope.
