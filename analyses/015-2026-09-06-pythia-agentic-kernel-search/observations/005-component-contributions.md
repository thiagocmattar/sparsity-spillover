# O005: sparse projection and FFN eligibility have different end-to-end effects

## Question

For the frozen 14M and 70M policies, which part of the eligible linear path is
associated with the end-to-end effect: FFN linears, attention projection
linears, or their combined full policy?

## Method and coverage

Run 025 measured 40 complete component probes: 20 on RTX PRO 4500 and the same
20 on H100 NVL. Each component probe changes only the eligible sparse linears
while retaining the same checkpoint, frozen implementation, workload, and
native reference:

- FFN-only uses `m`/`h` or the subset present in the frozen policy;
- attention-projection-only uses eligible `a`/`z` linears or their frozen
  subset;
- full policy uses the architecture's frozen winner.

Component points are single-process paired geometric means over 80 timings and
16 inputs; their bars are 95% input-cluster bootstrap intervals. Full-policy
points are the fresh-process medians from Observation O003; their bars are
process ranges. Every component probe passes complete 338-block validation.

## Result

All 18 attention-projection-only probes exceed native break-even: nine on each
GPU, with speedups from 1.0034x to 1.0254x. FFN-only crosses break-even in 11 of
22 probes: six on RTX and five on H100, over the wider range 0.8497x to
1.0246x.

The components are not additive. In the most conspicuous case, 70M A4-OL1 at
`kappa=0` has attention-projection-only speedup 1.0081x and FFN-only speedup
0.9545x on RTX, while the full policy falls to 0.7515x; the matched H100 values
are 1.0236x, 0.9109x, and 0.8539x. Expanding eligibility can therefore add
enough packing, dispatch, and fragmented-execution overhead to erase a small
component gain.

Conversely, for high-sparsity 14M A4/A7 points, both component paths are near
or above break-even and their full policies reach about 1.05x on RTX and
1.06--1.07x on H100. At 70M `kappa=0.5`, attention projection remains slightly
positive while the FFN path is GPU- and topology-dependent.

## Figure caption / legend

**Figure 5.** Full-model latency under FFN-only, attention-projection-only, and
combined frozen-policy eligibility. Lines match the same checkpoint across
ablations and do not imply additive effects. Component bars resample timing
inputs; full-policy bars span fresh processes. The dashed line marks native
break-even.

## Caveats

- "Attention projection" is not sparse attention. QK score and PV products
  remain dense SDPA; only `a`/`z` projection linears are eligible.
- Each component has one process, so its interval captures input variation but
  not process-to-process variation.
- A component probe changes the number and placement of wrapped modules, not
  just an abstract operation class. It diagnoses the integrated path and does
  not isolate one CUDA instruction-level cause.
- K010 at 410M is already a `z`-projection-only winner, so a separate 410M
  projection ablation would be identical to the full policy and was omitted.
- Exact `R_covered`, packing bandwidth, and dispatch counts were not serialized;
  the overhead interpretation is consistent with the ablation but not a direct
  counter-based decomposition.

## Sources

- Figure: [`figures/05-component-contributions.pdf`](../figures/05-component-contributions.pdf)
- Complete component table: [`component-results.csv`](../component-results.csv)
- Figure generator: [`04_plot_replications.py`](../04_plot_replications.py)
- Reduction: [`03_reduce_replications.py`](../03_reduce_replications.py)
