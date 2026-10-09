# Run069 observations

- [01: qualified latency boundaries](01-qualified-boundaries.md): the best
  no-skip penalty shrinks to 27-28%; sparse full-forward gains are 1.86-3.05%
  against matched gated native execution, with unchanged quality.
- [02: exploratory implementation screen](02-exploratory-screen.md): two-row
  reuse helps; direct stores are detrimental; prefetch does not close the dense gap.

The [diagnostic](../results/DIAGNOSTIC.md) combines the evidence and limitations.
Nsight hardware counters were denied. No practical global frontier win is established.
