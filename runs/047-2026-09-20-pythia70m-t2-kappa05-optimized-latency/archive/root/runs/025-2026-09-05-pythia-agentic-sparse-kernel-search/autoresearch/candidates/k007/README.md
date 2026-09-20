# K007: bounded tensor-core launch-geometry search

K007 preserves K004's exact 16-bit zero predicate, BF16 operands, FP32
accumulation, bias, and inline skip branch.  It changes only the compile-time
row/output/reduction tile geometry and launch warp count.  Six declared
configurations form the complete search space; runtime density does not select
a configuration.

The high-pressure 70M A4-OL1 and A7-OL1 development endpoints select a winning
configuration.  Any promoted implementation must be frozen in a new candidate
before held-out or complete-validation evaluation.
