# Narrative review v2: training models for sparse inference

Status: framing proposal for review. The TeX manuscript remains unchanged
pending approval.

## Paper thesis

> Training-time interventions across attention and MLP computations extend the
> validation-quality--activation-sparsity frontier beyond uniform magnitude
> clipping applied after training. A matched intervention ladder shows which
> architectural and optimization choices create the gain.

This is the primary contribution. **Model-wide activation sparsity
opportunity** provides the common evaluation axis.

## Motivation and literature gap

Activation sparsity matters because suitably organized zeros can reduce
inference time. A zero activation makes every scalar multiplication that
receives it redundant. The practical objective is to create many useful zeros
while preserving model quality and arranging them so sparse kernels can skip
the corresponding work.

Prior work offers complementary routes toward this objective. Training-free
methods such as [TEAL](https://arxiv.org/abs/2408.14690) and
[CHESS](https://arxiv.org/abs/2409.01366) choose thresholds for pretrained
models; TEAL reports model-wide sparsity, and CHESS also reaches selected
attention projections. Training-time methods such as
[ProSparse](https://arxiv.org/abs/2402.13516),
[Learn To Be Efficient](https://arxiv.org/abs/2402.06126), and
  [Sparser, Faster, Lighter Transformer Language Models](https://arxiv.org/abs/2603.23198)
primarily shape FFN activations. [Spark
Transformer](https://arxiv.org/abs/2506.06644) trains sparsity into both FFN and
attention computations.

These studies change architectures, training regimes, intervention sites,
optimization methods, and sparsity measures together. This leaves a comparative
gap: how does each design choice move a common quality--sparsity frontier when
the model, data, training budget, and evaluation remain matched?

## Scientific question

> Can progressively broader training-time sparsity interventions improve upon
> uniform evaluation-time clipping, which choices produce the improvement, and
> does the resulting recipe persist from 14M to 70M parameters?

## Main contribution: a matched intervention ladder

The ladder adds or replaces one component wherever a matched contrast is
available:

| Rung | Intervention | Question |
| --- | --- | --- |
| A0 | Stock GeLU Transformer | Dense reference |
| Clipped controls | Calibrated magnitude thresholds at four activations during evaluation | Frontier of a fixed trained model |
| A1-H | ReLU at the FFN hidden activation | Effect of an intrinsically sparse activation |
| A1-H-L1 | L1 pressure at the same activation | Effect of local sparsity pressure during training |
| A1-H-OL1 | Remove the component of the pressure update that opposes the optimizer's task update | Effect of conflict-aware pressure |
| A4 | Train with thresholds at the attention input, MLP input, MLP hidden activation, and attention context | Effect of distributing gates across attention and MLP components |
| A4-OL1 | Add pressure at those four activations | Interaction between placement and pressure |
| A7 | Add post-position-encoding query, key, and value thresholds | Effect of sparsifying attention operands |
| A7-OL1 | Add pressure at all seven activations | Complete architecture-wide recipe |

The full ladder was executed at 14M. The 70M study promotes A4-OL1 and A7-OL1
with the GeLU and ReLU controls. The 14M experiment provides rung-by-rung
attribution; the 70M experiment tests whether the selected recipes persist.

For a specified model and sequence workload, **model-wide activation sparsity
opportunity** is the fraction of scalar multiplications guaranteed to produce
zero because a measured activation supplied to them is exactly zero. The count
covers the attention and MLP projections, causal query-key and
probability-value products, and a dense language-model head. The head adds its
full multiplication count to the denominator. Kernel benchmarks then measure
how much of this arithmetic opportunity becomes inference speed.

## Main results

### Trained interventions extend the clipping frontier

The trained recipes achieve substantially better validation quality at similar
model-wide opportunity:

| Model | Trained intervention | Best evaluated clipped control nearby |
| --- | --- | --- |
| Pythia-14M | A7-OL1, threshold 0.1: loss **5.429** at **11.80%** opportunity | A0, target 0.9: loss **8.826** at **11.94%** |
| Pythia-70M | A4-OL1, threshold 0.1: loss **4.944** at **30.57%** opportunity | A1-H, target 0.8: loss **6.912** at **31.04%** |

The seven-site recipe also extends the attainable range. Its strongest setting
reaches 27.48% opportunity at loss 5.829 for 14M and 40.60% at loss 5.216 for
70M. The corresponding clipping grids reach at most 11.94% and 34.58%, at
losses 8.826 and 9.179.

### The 14M ladder locates the gain

The early rungs establish the FFN baseline. ReLU reaches 2.71% opportunity at
loss 5.270. Ordinary L1 reaches 3.95% at loss 5.102, while the matched
orthogonal-L1 endpoint reaches 3.94% at loss 5.121. This close comparison places
orthogonal L1 in the ablation study.

At threshold 0.1, expanding A4 to A7 adds query, key, and value thresholds and
raises opportunity from 8.95% to 10.43% for a loss change of 0.009. Adding
seven-site pressure raises opportunity again to 11.80% for a loss change below
0.001. At threshold 0.5, the same pressure adds 12.10 percentage points for a
loss change of 0.126. Query, key, and value zeros are valuable because each one
participates in many causal token-pair multiplications.

### The result persists at 70M

Both selected recipe families contribute points to the combined 70M frontier
and reach higher absolute opportunity than at 14M. The strongest A7-OL1 setting
rises from 27.48% to 40.60%, while its within-recipe loss increase falls from
0.349 to 0.275. For A4-OL1, the opportunity gain from its weakest to strongest
threshold rises from 4.90 to 9.92 percentage points at a loss increase of about
0.58 at either size.

The safe conclusion is persistence plus higher absolute opportunity at 70M.
The size effect varies by recipe and operating point. A scaling law requires
additional sizes, seeds, and matched training maturity.

## Treatment of 410M

Recommendation: place 410M in a secondary results paragraph and the appendix,
outside the abstract and the 14M--70M claim.

The common 1.493-billion-token budget gives 410M only 3.7 tokens per parameter,
compared with 21.2 at 70M and 106.1 at 14M. Its GeLU control loss of 4.547 is
weaker than the 70M control loss of 4.100, and its trained threshold response
changes direction. A7-OL1 reaches 80.62% opportunity at loss 5.121. This pattern
is consistent with insufficient training exposure. The learning-rate screen
leaves the baseline essentially unchanged; a longer-training comparison is the
direct test of the exposure explanation.

## Implications and new interventions

Sparse inference is a joint architecture--optimization--implementation
problem: training determines where zeros appear, the architecture determines
how often those values are reused, and the implementation determines which
patterns can be skipped efficiently.

The ladder provides a controlled way to add new intervention classes. These
directions have related precedents; their value here comes from measuring each
new rung against the same frontier.

1. **Joint activation and weight sparsity.** For `XW`, a multiplication is
   redundant whenever either operand is zero. Add weight-mask pressure and
   optimize the union of activation and weight zeros, counting overlaps once.
2. **Hardware-aligned groups and blocks.** Apply pressure to channel groups,
   attention-head groups, or token-by-channel tiles aligned with kernel
   dimensions. Measure active tiles alongside active scalar values.
3. **Shared paths across related tokens.** Encourage a batch, context window,
   or learned token cluster to use the same channel groups. Shared masks can
   amortize mask construction and reuse packed weights.
4. **Persistent sparse paths.** Coordinate gates across consecutive
   projections and measure whether residual additions and normalization restore
   the suppressed groups.
5. **Kernel-cost-weighted pressure.** Allocate pressure using measured latency
   or energy so training favors the sites and structures with the greatest
   realized benefit.

## Paper scope

The main paper should contain the intervention ladder, the 14M attribution, the
14M--70M frontier result, the attention-reuse explanation, and the model-wide
metric definition. Orthogonal-L1 details, spillover diagnostics, analytic reach
ceilings, complete grids, and the 410M curve fit naturally in the appendix.

## Candidate title

**Training Across the Transformer: An Intervention Ladder for Activation
Sparsity**

## Revised abstract

Activation sparsity can accelerate Transformer inference when sparse kernels
skip multiplications that receive zero-valued inputs. Post-training magnitude
clipping exposes this tradeoff after a model has learned its representations,
while training-time sparsification can shape those representations for
efficient execution. Existing studies vary the training regime, sparsified
components, optimization method, and evaluation measure together, leaving the
contribution of each choice unclear. We introduce a matched intervention ladder
that progressively replaces GeLU with ReLU, adds activation-sparsity pressure,
controls conflict between task and sparsity updates, expands thresholds
across attention and MLP inputs and outputs, and finally includes the query,
key, and value operands of attention. We execute the full ladder in randomly
initialized Pythia-14M and promote selected architecture-wide recipes to
Pythia-70M, training every model for the same 1.493 billion MiniPile tokens. We
compare validation loss against model-wide activation sparsity opportunity,
the fraction of scalar multiplications that receive an exact zero activation.
At approximately 12% opportunity, the trained 14M model attains loss 5.43,
compared with 8.83 for uniform evaluation-time clipping; at approximately 31%,
the trained 70M model attains 4.94, compared with 6.91 for the best nearby
clipped control. The seven-site recipe increases the attainable opportunity
from 27.5% at 14M to 40.6% at 70M. Matched 14M ablations show how
attention-operand thresholds and all-site pressure extend the trained frontier.
These results show that sparse inference benefits from joint design of training
objectives, intervention placement, and model-level evaluation.

## Evidence and claim limits

All models use random initialization and one 1.493-billion-token MiniPile pass.
Each endpoint covers all 500 validation documents: 338 complete 2,048-token
blocks and 692,224 tokens, with the 1,444-token tail recorded separately. The
14M and 70M studies use one matched seed per size. TEAL's blockwise allocation
would strengthen the training-free baseline; further seeds and sizes would
support generalization; sparse-kernel evaluation would quantify realized
latency and energy.
