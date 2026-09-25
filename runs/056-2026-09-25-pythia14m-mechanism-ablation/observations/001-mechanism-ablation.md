# Tile bypass and short-row execution at 14M

Question: how much do the two h/z execution mechanisms help conditionally and jointly?

Method: approved B/S factorial plus untouched K050, fixed T7/Pall kappa=0.5
checkpoint and gates, one RTX5090, BF16 B1/T2048 full-logit inference. Fifteen
fresh processes; 64 fixed timing inputs, seven paired passes. All 338 validation
blocks from 500 documents; 1444-token tail excluded. Full logits and h/z operands
match frozen K050 bitwise; all numerical gates and work-count checks pass.

Caption: (a) Full-model latency of four execution combinations and frozen K050.
(b) Conditional tile and short-row effects, joint savings and interaction.
Bars use geometric-mean host latency. Whiskers show process extrema and derived
difference spans, not confidence intervals. Positive values denote savings;
the interaction is not a third independently measured source of saved time.

Result: tile given short rows = +82.851 us;
short rows given tile = -1.661 us;
joint = +165.031 us;
interaction = -83.841 us.
See [the complete table](../results/mechanism-effects.md) and
[audited measurements](../results/mechanism-latency.json) for signs and variability.

Caveats: one checkpoint/device/workload. Whole-group completion belongs to the
short-row mechanism, including bias-only zero rows. Shared inspection, scheduling,
and padded fallback costs remain in the net effects. These effects do not uniquely
partition total savings, memory time, arithmetic time, or the PyTorch gap.

Source script: `13_report.py`; measurement audit: `07_reduce.py`.
