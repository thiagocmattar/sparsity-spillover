# Why seven-site attention zeros do not yet give K050 a net speedup

**The zeros make arithmetic unnecessary. K050 skips some of that arithmetic,
but the measured attention path is still slower after paying for the checks
and control needed to skip it.** There are three distinct quantities:

```text
Zero scalar products       Matrix-multiply instructions skipped       Time saved
       |                                  |                               |
 Mathematical opportunity       Depends on zero structure        Depends on all costs
```

The diagrams below are illustrative, not measured activation layouts.
`x` means a nonzero value; `0` means an exact zero. All toy values are finite.

## 1. A small Q/K example: seven of eight products are unnecessary

Take two queries and two keys, each with two features. Omit attention scaling
and the causal mask for this first example.

```text
        features                  features
Q = [ 1  0 ]              K = [ 1  0 ]
    [ 0  0 ]                  [ 0  2 ]

                         keys
Q K^T = [ 1*1 + 0*0     1*0 + 0*2 ] = [ 1  0 ]
        [ 0*1 + 0*0     0*0 + 0*2 ]   [ 0  0 ]
```

There are eight scalar multiplications. Seven have a zero operand; only
`1*1` is nonzero. A dense matrix multiplication still executes all eight.
An ideal implementation that skips individual zero products would execute
only one multiplication, although finding that product would also have a cost.

Now imagine one hardware instruction computes this entire tiny product.
Both input blocks contain a nonzero value. A kernel that can only skip an
instruction when **one complete input block is zero** must execute it.

Even completely zero dot products need not satisfy that condition:

```text
query: [ x  x  0  0 ]
key:   [ 0  0  x  x ]
        |  |  |  |
terms: [ 0  0  0  0 ]  --> dot product = 0

Neither input vector is entirely zero.
K050's whole-fragment check does not detect this disjoint-support case.
```

The conclusion is not that zeros cannot remove arithmetic. It is that this
kernel's skip rule cannot exploit every mathematically unnecessary product.

## 2. The same zero percentage can expose different skippable work

Each boxed group below represents one tiny Q operand fragment. Assume its
paired K fragment contains nonzeros. Each matrix has 32 entries, eight nonzero
and 24 zero: **75% sparsity in both cases**.

```text
Scattered nonzeros:                Nonzeros concentrated in one fragment:

     fragment 1    fragment 2            fragment 1    fragment 2
    +---------+  +---------+            +---------+  +---------+
    | x 0 0 0 |  | x 0 0 0 |           | x x 0 0 |  | 0 0 0 0 |
    | 0 x 0 0 |  | 0 x 0 0 |           | x x 0 0 |  | 0 0 0 0 |
    | 0 0 x 0 |  | 0 0 x 0 |           | x x 0 0 |  | 0 0 0 0 |
    | 0 0 0 x |  | 0 0 0 x |           | x x 0 0 |  | 0 0 0 0 |
    +---------+  +---------+            +---------+  +---------+
       RUN          RUN                   RUN          SKIP

       0 of 2 fragments empty              1 of 2 fragments empty
```

The right-hand pattern permits skipping the matrix-multiply contribution
associated with the empty fragment. It does not imply that the final score
tile is zero: other feature fragments may still contribute to that tile.

K050's actual attention instruction computes a larger fragment product:

```text
Q fragment                 K^T fragment                 score contribution
16 queries x 16 features   16 features x 8 keys        16 queries x 8 keys

        [ ... ]       @          [ ... ]        -->          [ ... ]

Entire Q fragment zero? ---- yes --> SKIP this instruction
Entire K fragment zero? ---- yes --> SKIP this instruction
Otherwise:                          RUN this instruction
```

This is a `16 x 8 x 16` matrix multiply-accumulate (MMA) instruction. **Either
operand being entirely zero is sufficient; both need not be zero.** Isolated
zero entries or an isolated zero query row inside an otherwise nonzero fragment
are insufficient for this rule. PV uses the analogous check on probability
and V fragments.

## 3. A zero attention score is not a zero attention probability

Applying row-wise softmax to the small example gives:

```text
Scores Q K^T                 Probabilities P = softmax(scores)

[ 1  0 ]         -->         [ 0.731  0.269 ]
[ 0  0 ]                     [ 0.500  0.500 ]
```

The zero score contributes `exp(0) = 1` to the softmax normalization.
It is not a masked connection, which would have score `-infinity` and zero
probability. With a causal mask, the zero query is uniform over its allowed
keys; future keys remain masked.

For example, using one value feature per key:

```text
V = [ 10 ]
    [ 20 ]

P V = [ 0.731*10 + 0.269*20 ] = [ 12.689... ]
      [ 0.500*10 + 0.500*20 ]   [ 15.000... ]
```

The displayed probabilities are rounded; the output uses unrounded softmax.
Setting the first row's zero-score connection to zero probability would
change the model's output. Thus Q/K zeros alone do not remove softmax or PV
work. V zeros can expose separate PV skipping opportunities, subject to the
same fragment-size constraint.

## 4. Even successful instruction skipping need not save time

The current attention helper loads fragments into registers before checking
their contents. Its execution is approximately:

```text
Attention skipping disabled:
    load fragments --> matrix multiplication --> remaining attention work

Attention skipping enabled:
    load fragments --> inspect fragments --> coordinate/branch
                                                |
                                  +-------------+-------------+
                                  |                           |
                              nonzero                       empty
                                  |                           |
                         matrix multiplication            skip it
                                  |                           |
                                  +-------------+-------------+
                                                |
                                     remaining attention work
```

The check does not automatically remove those fragment loads or the
surrounding normalization and output processing. Skipping helps only when
the runtime benefit of the omitted work exceeds the added execution cost.
The fraction of MMA instructions skipped is therefore not the fraction of
attention time saved.

We have measured the net outcome, not a separate timing for every check,
load, synchronization or arithmetic instruction. Detailed profiling is
still needed to determine which overhead dominates.

## 5. What happens in the actual seven-site checkpoint

For **Pythia-14M, seven-site + OL1, kappa = 0.5 (c30)**:

| Measurement | Result |
|---|---:|
| Eligible QK MMA instructions skipped | 58.00% |
| Eligible PV MMA instructions skipped | 67.21% |
| Full-model latency, projection skipping on, attention skipping off | 468.37 microseconds |
| Full-model latency, projection skipping on, attention skipping on | 473.37 microseconds |
| Net latency change from attention skipping | **+5.00 microseconds** |

**Arithmetic really is skipped here. The measured execution is nevertheless
slower.** This is distinct from the scattered-zero example, where the fragment
rule cannot skip the instruction in the first place. Both limitations matter.

The instruction counters cover all 338 validation blocks and include
causal/padded instruction opportunities. Timings are geometric means over
64 selected inputs x seven passes x three processes. They use the same
checkpoint, BF16 batch-one 2,048-token inference and hardware. The percentages
are not unmasked scalar-product fractions or percentages of total runtime.

This establishes a limitation of the tested K050 attention path at these
shapes, not a proof that attention sparsity cannot be profitable. A different
implementation might reuse zero metadata or avoid larger units of work;
that possibility has not been demonstrated by these measurements.

## Evidence and scope

This is an explanatory note using retained measurements and schematic
examples. It adds no experiment, result-bearing manuscript change or new
kernel claim.

- [Investigation report](investigation/README.md): c20/c30 comparison and
  per-checkpoint conclusions.
- [Kernel predicates and timing definitions](investigation/METHODS.md):
  fragment dimensions, pooling, counter coverage and attribution limits.
- [Checkpoint data](investigation/data/checkpoints.json): c30 instruction
  fractions and matched absolute latencies.
- [Attention helper source](../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k035/sparse_gemm.h):
  fragment copies, `run028_nonzero` and conditional MMA dispatch.
