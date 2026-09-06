# O004: frozen sparse policies do not preserve their value across GPUs

## Question

When checkpoint, implementation, batch, sequence length, precision, and
canonical logical counts are fixed, does relative sparse speedup remain stable
between RTX PRO 4500 Blackwell and H100 NVL Hopper?

## Method and coverage

The comparison contains the same 18 preregistered endpoint sentinels on each
GPU: A0, A1-H, and A4-OL1/A7-OL1 at `kappa` 0 and 0.5 for each of 14M, 70M,
and 410M. The policy is frozen as K013/K016/K010 respectively. RTX values are
medians of three eager-only processes; H100 values are medians of eager-only
repeats 2 and 3. Each process uses 80 paired timings over the same 16 inputs and
passes complete 338-block validation. All checkpoint hashes, canonical losses,
`R_model` numerators and denominators, and input identities reconcile across
the two hardware realizations.

Speedup is always native eager time divided by candidate eager time on the
same GPU. The slopes joining GPUs are not raw latency comparisons between an
RTX and H100.

## Result

| size | H100 faster than its RTX speedup | H100 slower | mean H100-minus-RTX speedup | observed range | RTX points above 1x | H100 points above 1x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14M | 4 / 6 | 2 / 6 | -0.0104x | -0.1321 to +0.0236x | 4 | 4 |
| 70M | 3 / 6 | 3 / 6 | +0.0133x | -0.0264 to +0.1024x | 3 | 2 |
| 410M | 0 / 6 | 6 / 6 | -0.0593x | -0.0944 to -0.0197x | 2 | 0 |

The transfer is neither a uniform hardware improvement nor a uniform
regression. H100 raises the four sparse 14M A4/A7 endpoints while sharply
worsening 14M A0. At 70M the direction splits evenly by checkpoint. At 410M,
all six policies lose relative value on H100 and the two Blackwell break-even
points disappear. This directly supports the scoped statement that kernel
value depends on GPU architecture and model shape.

## Figure caption / legend

**Figure 4.** Matched transfer of frozen policies from RTX PRO 4500 to H100
NVL. Each line joins the same checkpoint and implementation. Colour and marker
identify one of the six endpoint conditions. Bars span fresh-process speedup
estimates. The dashed line is native break-even on each GPU.

## Caveats

- Only two GPU architectures and one physical deployment per attempt were
  measured; this is not a general hardware ranking.
- Relative speedup can change because either native or candidate execution
  changes. The figure does not claim that the RTX has lower absolute latency.
- H100 uses two primary eager-only processes versus three on RTX because its
  first process was intentionally coupled to a CUDA-graph control.
- Hardware and software stack are bundled in this transfer: GPU architecture,
  compiler/runtime, and host environment are not isolated factors.
- QK/PV remain dense, and exact executed opportunity (`R_covered`) is absent.

## Sources

- Figure: [`figures/04-matched-hardware-transfer.pdf`](../figures/04-matched-hardware-transfer.pdf)
- Machine-readable pairs: [`hardware-transfer.csv`](../hardware-transfer.csv)
- Figure generator: [`04_plot_replications.py`](../04_plot_replications.py)
- Reduction: [`03_reduce_replications.py`](../03_reduce_replications.py)
