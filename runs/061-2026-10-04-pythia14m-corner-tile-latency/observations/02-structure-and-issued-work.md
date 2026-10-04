# How does tile structure relate to executed work?

Question: do the retained BF16 operands expose empty8x16 tiles and fewer issued
h/z MMA atoms alongside the measured latency difference?

Method: the untimed diagnostic pass reconstructs the exact signed operands
of each sparse candidate, counts exact/near-zero values, RMS/L2, empty8x16
tiles and row occupancy, and independently checks instrumented hybrid counters.
Coverage is all338 complete blocks/500 documents and all six layers,692224
inputs,1444 excluded tail tokens. Pool integer counts before dividing.

The [wide table](../results/diagnostic-tables.md) places site exact-zero and
empty-tile percentages side by side. A/M have zero exact zeros/empty tiles in
all four conditions. At κ=0, H empty tiles rise3.036% to57.515%, and Z.015%
to44.925%; at.1, H43.673% to84.362%, Z20.687% to70.974%. Element sparsity falls.
For tile κ=0, the consumer issues42.485%/55.075% of padded H/Z MMA potential;
for.1,15.638%/29.026%. This confirms changed structured support and work.

The [latency observation](01-corner-tile-latency.md) reports the accompanying
full-model tile/scalar ratios1.123472x/1.111937x. Tile checkpoints have worse
quality; fewer issued MMA atoms do not give a proportional full-model speedup.
The no-skip control disables MMA bypass and short-row SIMT substitution jointly.
Counters include row padding and substitution, with separate integer SIMT
products in the machine summary. They neither measure DRAM bytes nor isolate
H versus Z causal attribution; BF16 logical counts are explicitly a lower
bound distinct from canonical FP16 R_model. No decoding/scaling conclusion.

Sources: [generating script](../22_diagnostics_table.py),
[machine summary and input hashes](../results/diagnostic-summary.json),
[diagnostic implementation](../diagnostics.py). No additional figure is created.
