# T2/Ph in manuscript Table 1

The user requested T2 in the threshold/pressure definition table as well as
the result displays. The new row is T2/Ph: one-sided thresholds at h,z and
OL1 pressure at h. The caption now includes T2 in the five-value kappa sweep.

## Method and coverage

[05_table1_ceilings.py](../05_table1_ceilings.py) derives analytic reach from
the retained architecture counts for 14M, 70M and 410M. With h,z selected,
reachable operations are MLP W2 and the attention output projection:
L*T*(d_f*d+d*d) scalar products per full uncached 2,048-token sequence.
The denominator includes all six block operations, valid causal attention
pairs and the dense 50,304-token LM head. An independent dimension-based
formula checks numerator and denominator; the 14M fraction also matches the
retained Figure 1 ceiling exactly.

[Integer counts and source hashes](../data/table1-t2-ceilings.json) retain
the architecture dimensions, per-operation counts, reachable operations,
fractions and percentages. This is the manuscript's h,z topology, not the
operational registry's A2=m,h topology.

## Result and caption

Table 1 lists T2/Ph with threshold sites h,z, pressure site h, and ceilings
5.35%, 15.44% and 31.16% for 14M, 70M and 410M. These are analytic ceilings,
not observed sparsity or measured speedups; a ceiling entry does not claim
that a training run was completed at that size. All existing rows remain
unchanged.

## Verification

The rebuilt PDF remains 21 pages. Only page 4's extracted text changes;
Table 1 and the following architecture figure fit without overlap or
clipping and were visually checked. References and citations resolve, with
no overfull boxes and the same three pre-existing underfull warnings.
See [Table 1 verification](../data/table1-verification.json). The author's
other TeX edits are preserved, and only the targeted table additions are
staged from methodology.tex.
