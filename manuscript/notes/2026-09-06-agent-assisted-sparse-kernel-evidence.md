# Agent-assisted sparse-kernel evidence

- Date: 2026-09-06
- Status: paper-planning note; not result-bearing manuscript text
- Evidence: Run 025 and Analysis 015

This note records the argument that the completed bounded systems study can
support later in the paper. It does not promote a research finding, revise TeX,
or turn an exploratory comparison into a confirmatory experiment.

## Recommended argument chain

1. Canonical `R_model` measures model-wide logical zero-product opportunity;
   it is not an eliminated-FLOP count or runtime speedup.
2. The official Sakana kernel is a useful starting point, but its benefit does
   not transfer mechanically across Pythia widths, GPU architectures, or site
   policies. Packing, dispatch, shape, and coverage overhead can dominate.
3. A bounded Sakana-derived, agent-assisted implementation search produced
   qualified end-to-end speedups for every studied Pythia architecture on at
   least one GPU: 1.0725x at 14M, 1.0452x at 70M, and 1.0192x at 410M.
4. The fresh-process `R_model` association is real but conditional: it is
   strong at 410M, moderate at 14M, and weak/nonlinear at 70M. A positive slope
   can also order implementations that all remain below native break-even.
5. At identical checkpoints and canonical `R_model`, targeted implementation
   changes improve the architecture-level median throughput ratio by 1.0152x,
   1.0113x, and 1.0064x at 14M, 70M, and 410M respectively; 16/18 fresh process
   pairs favor the optimized policy.
6. A broader 14M policy is 5--7% slower than P0 on the same endpoints. Kernel
   specialization therefore carries measurable tuning risk as well as upside.

The strongest defensible conclusion is a scoped systems case study: logical
opportunity helps identify promising sparse checkpoints, but hardware- and
shape-specific implementation search is required to convert some of that
opportunity into wall-clock benefit.

## Evidence table

| Claim component | Direct evidence | Boundary |
| --- | --- | --- |
| Speedup exists at 14M/70M/410M | Best qualified fresh-process medians of 1.0725x, 1.0452x, and 1.0192x | The winning GPU differs; gains are batch-one prefill results |
| `R_model` associates with speedup | RTX OLS R2 0.347/0.0027/0.636 and H100 OLS R2 0.321/0.051/0.700 for 14M/70M/410M | 70M is near-null; H100 410M remains below break-even |
| Hardware matters | Matched H100-minus-RTX mean speedup changes of -0.0104x, +0.0133x, and -0.0593x | Two GPU architectures and bundled software stacks |
| Optimization matters at fixed `R_model` | Architecture medians 1.0152x/1.0113x/1.0064x; 16/18 pair wins | 14M P0-to-K001 confirmation is adaptive |
| Bad specialization adds overhead | K013/P0 0.9370x and 0.9477x on 14M high-`kappa` endpoints | K013 also repaired numerical validity elsewhere |
| Attention and FFN paths differ | 18/18 attention-projection-only probes versus 11/22 FFN-only probes above break-even | QK/PV remain dense; components are not additive |

## Claims to avoid

- Do not say `R_model` predicts a universal or monotonic speedup.
- Do not equate high OLS R2 with useful acceleration.
- Do not claim sparse QK/PV attention; only attention projection linears were
  specialized.
- Do not claim agent superiority over human or non-agent optimizers. There is
  one logged search trajectory and no matched optimizer control.
- Do not imply multi-seed training replication. Independent processes measure
  systems variance only.
- Do not hide failed numerical deployments or the K013 negative result.
- Do not call native eager the strongest possible dense implementation.
- Do not generalize the Sakana-derived runtime result to TEAL kernels. This
  study did not benchmark a TEAL runtime implementation.

## Suggested paper placement

Use this as a compact systems subsection after the `R_model` frontier results,
not as the central training contribution. A main figure can pair the
fresh-process `R_model` panels with the fixed-`R_model` implementation figure.
The matched GPU-transfer and component figures fit naturally in an appendix or
systems supplement.

Suggested language:

> Logical sparsity is an opportunity, not a speed guarantee. In a bounded
> Sakana-derived implementation search, the conversion from canonical
> zero-product opportunity to end-to-end throughput depended strongly on model
> width, GPU architecture, and sparse coverage policy. Holding checkpoint and
> `R_model` fixed, specialized policies yielded small but repeatable median
> gains at all three widths, while an over-broad policy produced a measurable
> slowdown.

## Provenance

- `runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/README.md`
- `analyses/015-2026-09-06-pythia-agentic-kernel-search/README.md`
- `analyses/015-2026-09-06-pythia-agentic-kernel-search/observations/003-fresh-process-rmodel-hardware.md`
- `analyses/015-2026-09-06-pythia-agentic-kernel-search/observations/004-matched-hardware-transfer.md`
- `analyses/015-2026-09-06-pythia-agentic-kernel-search/observations/005-component-contributions.md`
- `analyses/015-2026-09-06-pythia-agentic-kernel-search/observations/006-fixed-rmodel-kernel-optimization.md`
