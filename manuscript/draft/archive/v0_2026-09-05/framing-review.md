# Critical review of the post-410M paper framing

Reviewed source:
`manuscript/notes/2026-09-04-paper-framing-after-pythia410m.md`.

## Overall judgment

The note makes the right high-level decision: do not hide or rerun away the
410M result, and do not let it erase the clean evidence that exists at smaller
scales. Its strongest move is to treat the 410M cohort as a boundary on
portability under the executed fixed-token protocol. The three-panel absolute
figure plus within-size deltas is also the right visual strategy.

The framing is not yet submission-safe. It sometimes moves too quickly from an
observed regime difference to an optimization explanation, overstates the
novelty gap relative to TEAL's model-level sparsity, and calls complete tested
curves “frontiers” even when some points are dominated. The paper should center
the exact accounting problem and present the intervention results as a
carefully delimited empirical case study.

## Likely reviewer criticisms and repairs

### 1. “Model-wide sparsity already exists in TEAL; what is new?”

TEAL defines a model-level configuration and weights per-matrix sparsity by
matrix footprint. A reviewer can therefore reject any claim that prior work
reports only local sparsity. The defensible gap is narrower: the present work
counts actual exact-zero scalar products over a declared full-sequence causal
graph, including post-RoPE QK operands, valid-causal QK/PV reuse, the actual
pre-`W_o` context, and a dense LM-head denominator. The related-work and
introduction sections must compare these definitions explicitly.

**Repair:** claim topology-aware, operand-level accounting for a complete
declared workload, not the first “model-wide sparsity” metric.

### 2. “Removable computation is not removed computation.”

`R_model` assigns one unit to each scalar multiplication with a zero operand.
Irregular masks, index overhead, packing, memory traffic, kernel granularity,
and dense fallback can prevent those products from being skipped. Run 022 does
not supply a successful runtime result.

**Repair:** use “logical zero-product opportunity” throughout. Reserve
“removed,” “saved,” “speedup,” and “efficient” for future kernel measurements.
Include the failed raw-ELL compatibility attempt only as engineering provenance,
not scientific evidence.

### 3. “`R_model_max` is not a maximum of measured `R_model`.”

The measured numerator includes natural zeros outside the selected topology,
whereas `R_model_max` counts operations reachable when selected sites are all
zero. This is why `U_arch` is not guaranteed to be at most one. Calling the
quantity a ceiling without this qualification invites a formal objection.

**Repair:** introduce it as *selected-site analytic reach* and retain
`R_model_max` as the symbol for continuity. State immediately that it is not an
upper bound on total measured zeros from other sites. Avoid `U_arch` in the main
paper.

### 4. “The inference workload does not match modern decoding.”

The denominator is an uncached, full-sequence, length-2048 causal forward pass.
Autoregressive decoding with a KV cache has different QK/PV reuse and often a
memory-bound kernel regime. The current `R_model` values cannot be transferred
unchanged to decode latency.

**Repair:** call the workload full-sequence/prefill-like and keep sequence
length in the metric signature. Discuss cached decoding as a different required
accounting contract.

### 5. “One seed and one small corpus cannot support a general method claim.”

Every scale has one initialization/data-order seed and one MiniPile pass. The
largest model receives only 3.684 tokens per parameter. There are no downstream
tasks, no confidence intervals, and no independent corpus.

**Repair:** describe the experiment as a controlled mechanistic study and the
curves as descriptive. The highest-value confirmation is additional 14M seeds
for the clean A4/A7 contrast, followed by a predeclared longer-horizon A0 test
if 410M optimization becomes central.

### 6. “Calling 410M undertrained is post-hoc storytelling.”

The baseline inversion, continuing late loss decrease, clipping incidence, and
low token exposure make an insufficient horizon plausible. Run 021 rules out
only two higher peak learning rates. It does not test more tokens, a second
pass, a lower LR, or a different schedule.

**Repair:** report a fixed-token regime dependence and an unresolved horizon
hypothesis. Use “stress test” only to describe its role in the paper, not as
evidence that this role was preregistered.

### 7. “A4-OL1 versus A7-OL1 is confounded.”

At 70M and 410M the selected recipe contrast changes both the gate topology and
the OL1 pressure set. The high-dose A7 dominance cannot be attributed to the
three attention gates alone. The clean no-pressure A4/A7 topology contrast
exists only at 14M.

**Repair:** call A4-OL1 and A7-OL1 complete recipes. Use Finding F002 only for
the matched 14M A4/A7 topology result. Phrase cross-scale A7-over-A4 as recipe
ordering.

### 8. “The proposed method has weak comparative evidence.”

Operational OL1 enforces a local non-opposition geometry in AdamW direction
space, but the endpoint advantages over naive L1 are small and inconsistent.
Projection does not guarantee a finite step lowers task or validation loss.

**Repair:** present OL1 as an optimization mechanism and ablation, not a
generally superior method. Explain the trust bound and its active incidence;
retain full gradient trajectories in the supplement.

### 9. “Sparsity spillover is overnamed relative to the effect.”

The `v` near-zero mass increases with `h` pressure, Q/K/m move weakly and
non-monotonically, and the post-`W_o` response reverses at the largest dose.
This supports nonlocal site-specific redistribution, not a uniform opposing
attention response or a causal pathway.

**Repair:** keep “sparsity spillover” as secondary diagnostic language, define
it descriptively, and do not use it as the title-level claim. Show the actual
small percentages and separate the GeLU/ReLU contrast from the within-ReLU
lambda trajectory.

### 10. “The paper calls tested sets frontiers without computing the envelope.”

At 410M, high-dose points dominate several intermediate trained points. Lines
through every dose are dose-response curves, not all Pareto-optimal points.

**Repair:** call figures “quality--opportunity curves” or “evaluated sets” and
use “frontier” only for the nondominated envelope. Keep every dose visible to
avoid favorable endpoint selection.

### 11. “The attention accounting can be misunderstood.”

PRE-RoPE zeros do not necessarily survive rotation; probability zeros may arise
from floating-point underflow; V zeros have position-dependent reuse; and
future causal entries are excluded rather than credited. A marginal activation
percentage cannot reproduce these counts.

**Repair:** retain the exact operand equations in the main method and move the
indexed OR-count formulas to the appendix. Make the actual post-RoPE and
valid-causal contracts explicit in every figure caption.

### 12. “The result hierarchy reflects post-result selection.”

The selected ladder follows extensive 14M exploration, and the paper-facing
choice to emphasize A4/A7 was made after seeing outcomes. That does not
invalidate the evidence, but it makes confirmatory language inappropriate.

**Repair:** distinguish exploratory development, the predeclared 70M/410M
promotion, and the post-result paper synthesis. Provide a chronological evidence
ledger and do not present the full ladder as one preregistered experiment.

## Revised argument hierarchy

1. **Measurement contribution:** local activation sparsity is not invariant to
   site, operation, causal reuse, architecture, or workload; actual operand
   counts make those choices explicit.
2. **Diagnostic contribution:** the same pooled local sparsity can map to very
   different operation contributions, and pressure at one site can redistribute
   other activation marginals.
3. **Empirical case study:** architecture-wide trained recipes trace informative
   quality--opportunity curves at 14M and 70M.
4. **Boundary result:** under the equal-token 410M run, the baseline and trained
   dose response change qualitatively; a higher-LR screen does not repair the
   baseline.
5. **Open questions:** which effects replicate across seeds and horizons, how
   masks organize over tokens/heads/layers, and which logical patterns are
   exploitable by real kernels.

This hierarchy leaves the paper with useful negative evidence and more
questions than answers without making the argument vague: the exact accounting
is the stable center, while the intervention and scale results establish where
the empirical story currently holds and where it breaks.

