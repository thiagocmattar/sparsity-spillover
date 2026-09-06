# O006: specialized kernels improve speed at fixed logical opportunity

## Question

Can a bounded implementation search improve end-to-end sparse-model
throughput without changing the trained checkpoint or its canonical
`R_model`, and can a poorly chosen specialization make the same checkpoint
slower?

## Method and coverage

Run 025 attempt `rtxpro4500-004` measured A4-OL1 and A7-OL1 at `kappa=0.5`
for Pythia-14M, 70M, and 410M on the same RTX PRO 4500 Blackwell used for the
search. Each implementation/condition realization ran in a separate eager-only
Python process. Repeat order was counterbalanced. Every process used five
passes over 16 fixed sequence-length-2,048 timing inputs, producing 80 paired
native/candidate ratios, followed by complete numerical and loss validation on
all 338 MiniPile validation blocks (692,224 input tokens; 1,444-token tail
excluded).

Within each pair, checkpoint hashes, model weights, data, gate topology,
workload, and the integer numerator and denominator underlying canonical
`R_model` are identical. Only implementation coverage and dispatch policy
change. Three process pairs were run per condition.

The prespecified Phase-17 contrasts are P0 versus K013 at 14M, K009 versus K016
at 70M, and K004 versus K010 at 410M. Because K013 was slower than P0, the
direct P0-to-K001 14M contrast was added adaptively in Phase 18 and is the
primary positive 14M comparison. It is evidence from the search trajectory,
not a preregistered confirmatory test.

## Result

| Size | Condition | `R_model` | Baseline -> optimized | Median ratio | Process-pair range | Wins |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 14M | A4-OL1, `kappa=0.5` | 12.713% | P0 -> K001 | 1.0202x | 0.9932--1.0287x | 2/3 |
| 14M | A7-OL1, `kappa=0.5` | 27.483% | P0 -> K001 | 1.0102x | 1.0093--1.0218x | 3/3 |
| 70M | A4-OL1, `kappa=0.5` | 35.596% | K009 -> K016 | 1.0132x | 1.0093--1.0188x | 3/3 |
| 70M | A7-OL1, `kappa=0.5` | 40.602% | K009 -> K016 | 1.0075x | 0.9897--1.0869x | 2/3 |
| 410M | A4-OL1, `kappa=0.5` | 71.591% | K004 -> K010 | 1.0063x | 1.0037--1.0092x | 3/3 |
| 410M | A7-OL1, `kappa=0.5` | 80.616% | K004 -> K010 | 1.0065x | 1.0057--1.0113x | 3/3 |

Across the two conditions and three process pairs per architecture, the median
optimized-over-baseline ratio is 1.0152x at 14M, 1.0113x at 70M, and 1.0064x
at 410M. All three architecture medians exceed one, and 16/18 individual
process pairs favor the optimized policy. All 48 processes pass complete
validation.

The 14M final robustness policy supplies the adverse case. K013/P0 is 0.9370x
on A4-OL1 and 0.9477x on A7-OL1, with zero of six process pairs favoring K013.
K013 repaired numerical validity elsewhere in the full ladder, but its broader
coverage adds enough overhead to lose throughput on these endpoints.

The result directly supports an implementation-sensitivity claim: fixed
logical opportunity does not fix runtime, and targeted implementation changes
can recover small end-to-end gains. It does not show that every search endpoint
improves, nor that an agent is better than a human or another optimizer.

## Figure caption / legend

**Figure 6.** Fresh-process sparse-kernel optimization at fixed checkpoint and
canonical `R_model`. Each thin line joins baseline and optimized throughput in
one process pair; the large line joins process medians. Panels separate Pythia
architectures, and x-axis labels show the unchanged logical opportunity. The
14M panel uses the adaptive direct P0-to-K001 confirmation; the prespecified
K013 negative contrast is retained in the accompanying table.

## Caveats

- The 14M positive contrast is adaptive and should be labeled exploratory.
- Three process pairs capture launch/runtime variation, not training-seed
  uncertainty; each checkpoint still represents one trained seed.
- The improvements are small, especially at 410M, and are specific to this GPU,
  software stack, batch-one prefill workload, and timing protocol.
- Native eager is the reference. The strongest compiled-dense comparator was
  not completed.
- QK/PV attention products remain dense. Attention sparsity here means eligible
  projection linears, not a sparse SDPA kernel.
- Canonical `R_model` includes logical opportunities outside the linears that
  execute sparsely; exact executed opportunity was not serialized.

## Sources

- Figure: [`figures/06-fixed-rmodel-kernel-optimization.pdf`](../figures/06-fixed-rmodel-kernel-optimization.pdf)
- Complete tables: [`fixed-rmodel-tables.md`](../fixed-rmodel-tables.md)
- Process pairs: [`fixed-rmodel-process-pairs.csv`](../fixed-rmodel-process-pairs.csv)
- Reduction: [`05_reduce_fixed_rmodel.py`](../05_reduce_fixed_rmodel.py)
- Figure generator: [`06_plot_fixed_rmodel.py`](../06_plot_fixed_rmodel.py)
- Source attempt: [Run 025 attempt 004](../../../runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/launch-control/rtxpro4500-004/README.md)
