# Rewritten manuscript: argument and author-review guide

Working title: **Where to Sparsify a Transformer: Matched Interventions in
Architecture and Optimization**.

The full rewrite is in [main.pdf](main.pdf), with editable sections under
[main.tex](main.tex). It uses the strongest existing intervention comparisons,
adversarial-review-v3, and the now-completed 70M execution evidence.

## The main contribution

We provide a matched empirical account of how gate location and activation
pressure jointly change the quality-sparsity trade-off. The central result is
that extending the pressure sites can raise its quality cost, while adding
attention-operand gates opens additional opportunity and changes the response
to the expanded pressure recipe. The effects depend on dose and quality
allowance. R_model provides the common accounting needed to compare them.

This is a representation and optimization-architecture paper motivated by
efficient inference. Its positive result concerns what training makes sparse
and at what quality cost. The runtime study establishes a separate limitation
of one implementation.

## Motivation, question, and literature gap

Speedup is the motivation: useful zeros must occur in representations consumed
by operations that an implementation can exploit. The scientific question is
how the effect of pressure depends on the gate and pressure locations.

The literature already contains train-free clipping, sparse FFNs, all-linear
input sparsity, and sparse attention. In particular,
[Q-Sparse](https://arxiv.org/abs/2407.10969) and
[Spark Transformer](https://arxiv.org/abs/2506.06644) prevent a credible claim
that this is the first model-wide or integrated architecture/training approach.
The revised gap is narrower: matched comparisons of fixed-threshold gates,
explicit pressure sets, complete update recipes, and attention operands under
common initialization, training, and evaluation.

## The empirical argument

1. **Local pressure supplies a useful reference.** At 14M, both ordinary L1
   and OL1 improve on the ReLU control in parts of the grid. Their ordering is
   mixed, so OL1 remains an ablation.
2. **Broader pressure carries a quality cost.** Holding A4 gates fixed,
   expanding pressure from h to four sites raises loss at every threshold.
   At kappa=.01, opportunity remains about 8.59% while loss rises by .254.
3. **Attention gates change the available opportunity and pressure response.**
   At kappa=.5, Q/K/V gates alone add 5.17 percentage points with .043 added
   loss. Pressure adds 12.10 points and .127 loss under A7, versus 2.50 points
   and .378 loss under A4. This is an exploratory difference between two
   configuration-specific objectives; the fixed-pressure A7 cell is missing.
4. **Frontier gains depend on the permitted quality cost.** Training is
   stronger at every reported 14M allowance. At 70M, uniform clipping is
   stronger at tight allowances, while training extends the frontier at a
   larger allowance. The table shows the complete operating-point comparison.
5. **The operation decomposition explains both gains and limits.** A7 adds
   QK/PV opportunity. The higher 70M model-wide total also reflects the larger
   share of work in transformer blocks: A7 block-level opportunity actually
   falls from 91.75% to 82.15% between 14M and 70M at the largest threshold.

| Added cross-entropy | 14M trained | 14M clipping | 70M trained | 70M clipping |
|---:|---:|---:|---:|---:|
| .05 | 9.34% | 2.55% | 0.00% | 7.41% |
| .10 | 9.34% | 4.42% | 0.00% | 7.41% |
| .25 | 11.80% | 5.27% | 10.07% | 19.99% |
| .50 | 15.39% | 6.13% | 10.07% | 22.52% |
| 1.00 | 27.48% | 6.99% | 30.57% | 25.11% |

Values are maximum evaluated model-wide opportunity within each allowance,
relative to that size's A0 loss. The 0.00% entries are rounded natural-zero
opportunity of the eligible A0 control. An added loss of .05 permits about
5.1% higher perplexity; an added loss of 1 permits a 2.72x increase.
The 70M training grid is smaller than the detailed 14M grid.

## What changed beyond the review

Run 023 completed after the review's evidence cutoff. Its sparse Pythia path
passes correctness checks but runs slower than dense in every tested
configuration. The best full-model ratio is .983x at batch one. This becomes
a short main result and a complete appendix table, with the covered operations
and workload specified.

The high-threshold h and z tensors are more than 99.8% zero in the four main
14M/70M endpoints. The text reports this directly and identifies a
capacity-matched reduced-branch comparison as an informative control. High
opportunity should be read together with the quality loss and representation
suppression.

The 410M evidence is preserved in the appendix and disclosed in the main
discussion. Its token exposure is much lower per parameter, but the current
data do not diagnose undertraining. Its full frontier, within-family reversal,
and separate learning-rate screen remain visible.

## Implications

The study motivates objectives that specify both sparse representations and
the operations that consume them. Joint weight/activation pressure could
target complementary zeros, while blockwise or shared-channel pressure could
encourage paths reusable across tokens. These remain future interventions.
Post-training, including supervised or reinforcement-learning adaptation,
would assess whether learned sparsity survives a changed objective.

The most consequential remaining acceptance risks are independent-seed
replication, the missing fixed-pressure condition, and strong matched
baselines. The rewrite exposes the current evidence more effectively without
assigning a new numerical acceptance probability.
