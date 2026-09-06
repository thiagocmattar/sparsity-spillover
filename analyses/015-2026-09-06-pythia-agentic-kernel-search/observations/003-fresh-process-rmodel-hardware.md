# O003: the R_model-speed association is conditional on model shape and GPU

## Question

After freezing the Run 025 winner separately for Pythia-14M, 70M, and 410M,
does the association between canonical `R_model` and measured full-model
speedup survive fresh Python processes and transfer from Blackwell to Hopper?

## Method and coverage

- Frozen policies: K013 for 14M, K016 for 70M, and K010 for 410M. No H100
  result was used to retune or select them.
- Workload: batch-one, sequence-length 2,048 BF16 Pythia prefill with complete
  logits and the dense LM head. QK and PV remain dense SDPA.
- RTX PRO 4500 Blackwell: all 36 trained checkpoints, three eager-only Python
  processes per checkpoint.
- H100 NVL Hopper: the 18 preregistered endpoint sentinels, using eager-only
  repeats 2 and 3. Repeat 1 deliberately initialized CUDA graphs and is kept as
  a separate control rather than mixed into the primary eager estimate.
- Each process estimate is the paired geometric mean of 80 native/candidate
  host-time ratios over 16 fixed seed-2500 training-cache blocks. The plotted
  bar is the range across process estimates.
- Every primary process has complete numerical and loss validation on all 338
  complete MiniPile blocks: 500 documents, 692,224 input tokens, and the
  declared 1,444-token excluded tail.
- `R_model` is copied from and integer-reconciled against the checkpoint's
  complete canonical logical-product pass. Repeated timings do not create new
  training seeds.

The regression unit is one checkpoint's median process speedup. OLS includes an
intercept and excludes failed numerical deployments. Spearman correlation and
leave-one-checkpoint-out slope ranges are retained in
[`replication-regressions.csv`](../replication-regressions.csv).

## Result

| GPU | size | qualified / total | slope per +10 pp `R_model` | OLS R2 | Spearman rho | points above 1x | best qualified median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 4500 | 14M | 11 / 12 | +0.0347x | 0.347 | +0.573 | 9 | 1.0544x, A4-OL1 `kappa=0.01` |
| RTX PRO 4500 | 70M | 9 / 12 | +0.0036x | 0.0027 | +0.467 | 4 | 1.0331x, A4-OL1 `kappa=0.5` |
| RTX PRO 4500 | 410M | 12 / 12 | +0.0105x | 0.636 | +0.867 | 2 | 1.0192x, A4-OL1 `kappa=0.5` |
| H100 NVL | 14M | 6 / 6 | +0.0649x | 0.321 | +0.429 | 4 | 1.0725x, A7-OL1 `kappa=0` |
| H100 NVL | 70M | 6 / 6 | +0.0100x | 0.051 | +0.429 | 2 | 1.0452x, A4-OL1 `kappa=0.5` |
| H100 NVL | 410M | 6 / 6 | +0.0075x | 0.700 | +0.829 | 0 | 0.9326x, A7-OL1 `kappa=0.5` |

Every architecture has at least one qualified speedup on at least one GPU:
1.0725x for 14M and 1.0452x for 70M on H100, and 1.0192x for 410M on RTX.
However, the association is not universal. It is near-null for 70M on both
GPUs. The high H100 410M R2 orders six slow implementations and does not imply
that any crossed native break-even. Correlation and usable acceleration are
therefore separate findings.

The four Blackwell numerical failures replicate in all three fresh processes:
14M A7-OL1 at `kappa=0.1`, and 70M A4-OL1 at `kappa` 0.01, 0.05, and 0.1.
They remain plotted and are excluded from qualified fits.

## Figure caption / legend

**Figure 3.** Canonical logical-product opportunity versus frozen full-model
speedup on RTX PRO 4500 Blackwell and H100 NVL Hopper. Colour identifies model
size and marker identifies the trained family. Points are checkpoint-level
process medians; bars span independent process estimates. Red crosses mark
failed complete numerical gates. Lines and displayed R2/rho values are
descriptive within-size fits over qualified checkpoints only.

## Caveats

- This is one trained seed per checkpoint. Timing processes measure systems
  variability, not training uncertainty.
- RTX has the full 12-condition ladder per size; H100 has six endpoint
  sentinels per size. Their R2 values have different support.
- The 16 timing inputs are fixed training-cache blocks rather than the planned
  seed-2504 validation sample; complete 338-block quality validation is
  unaffected.
- `R_model` includes logical opportunities outside the linears that actually
  execute sparsely. Exact `R_covered` was not recorded, so this analysis must
  not attribute the slope to executed sparse coverage.
- No sparse QK/PV attention kernel is present. Attention contribution here is
  limited to eligible projection linears.
- A regression with positive slope may remain entirely below 1x, as the 410M
  H100 panel demonstrates.

## Sources

- Figure: [`figures/03-fresh-process-rmodel-vs-speedup.pdf`](../figures/03-fresh-process-rmodel-vs-speedup.pdf)
- Figure generator: [`04_plot_replications.py`](../04_plot_replications.py)
- Reduction: [`03_reduce_replications.py`](../03_reduce_replications.py)
- Complete tables: [`replication-tables.md`](../replication-tables.md)
- Source run: [Run 025](../../../runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/README.md)
