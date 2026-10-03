# Gates after LayerNorm and useful tile sparsity

The diagnostic supports two separate hypotheses: bilateral gates may retain
useful negative information at `a/m`, and explicitly structured training may
create more tiles that a kernel can skip. Neither hypothesis has been tested
directly by the retained training comparisons.

## Method and coverage

Source and generating script: [01_collect.py](../01_collect.py), with exact
source identities and counts in [diagnosis.json](../data/diagnosis.json).
This is an audit of completed runs, with full 338-block validation coverage.
No new model forward pass or latency measurement is performed. All training
comparisons are descriptive and use the existing single-seed cohort.

## The gate changes which information survives

The operational unilateral rule retains `x >= kappa`; the bilateral rule
retains `abs(x) >= kappa`. Thus the unilateral rule removes a large negative
activation even at kappa zero, while the bilateral zero-threshold gate is an
identity. The distinction applies to both values and gradients. A positive
bilateral threshold also masks gradients inside its rejected central band.

LayerNorm normalizes features within each token before its learned affine
scale and shift. It does not bound activations to minus one through one or
guarantee symmetry, a zero median, or a 50 percent negative fraction after
the affine transform. See [PyTorch LayerNorm](https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html).

The retained 14M baseline `m` distribution makes the user's intuition concrete:
265,352,753 of 531,628,032 elements are negative, approximately **49.91%**.
Only 3,843,865 elements, approximately **0.72%**, satisfy `abs(m) <= 0.01`.
Its pooled mean is approximately -0.000068, RMS 1.0148, and range
approximately -5.55 through 5.48. A unilateral zero-threshold gate on these
baseline operands would remove the negative half; this is not a measurement
of the adaptation caused by training with that gate. The near-zero counter
uses `<=`, whereas the bilateral gate rejects `<`; it is not an exact predicted
zero count for a bilateral kappa 0.01 intervention.

This baseline histogram contains `m` but not `a` or `z`. The trained signed
histograms are captured after gates, so their zero mass does not identify
the magnitudes or signs of values that were removed. Energy removed by the
trained unilateral gates and learned LayerNorm affine shifts remain unknown.

## Adding the studied a and m gates has a quality cost

T2/Ph and T4/Ph keep the same h/z gates and pressure at h. T4 adds unilateral
gates at a and m jointly. The 14M initial parameters, realized data order,
712 updates and 1,493,172,224 training tokens match in the retained records.

| Kappa | 14M loss T4 minus T2 | 70M loss T4 minus T2 |
|---|---:|---:|
| 0 | +0.087147 | +0.477820 |
| 0.01 | +0.073854 | +0.468628 |
| 0.05 | +0.061211 | +0.403853 |
| 0.1 | +0.077589 | +0.519653 |
| 0.5 | +0.186411 | +0.467676 |

Positive differences indicate worse validation loss. The 70M compact records
are imported from the verified historical collector; this audit does not
independently recheck their raw training matching. These comparisons isolate
the joint addition of the studied a/m gates, not a versus m separately, the
effect of LayerNorm, or whether bilateral gates perform better. They also do
not make discarded negative magnitude a demonstrated causal mechanism.

## Scalar zeros do not imply empty tiles

Run039 holds the 14M T7/Pall checkpoint at kappa 0.5 fixed and changes only
execution. Its BF16 operand counters show:

| Site | Exact zeros | Rows with at most two nonzeros | Empty 8 by 16 tiles |
|---|---:|---:|---:|
| a | 75.6292% | 15.3706% | 11.1218% |
| m | 68.6494% | 0.0000% | 0.0000% |
| h | 99.8755% | 92.7546% | Not reported in this reduction |
| z | 99.9170% | 99.7254% | Not reported in this reduction |

The table pools raw integer counts before division. Its rows do not describe
the baseline distribution or T2/Ph endpoints. The direct h/z-style port to a/m
also pads eight real rows into sixteen-row matrix work. Both ports together
take 0.715227 ms, versus 0.708533 ms for their layout-matched dense control
and 0.494995 ms for the frozen kernel. Most regression comes from that layout;
sparse handling adds 6.694 microseconds. This limits the interpretation of
that particular failed port and does not show that a/m acceleration is impossible.

Run037's conditional execution savings at a/m are -1.629/-2.361 microseconds,
with process-extrema spans crossing zero. Those small differences do not
establish a reliable standalone slowdown. Its h/z savings are positive with
spans excluding zero. These effects depend on the other execution paths and
are not additive allocations of total speedup.

Run056 supplies stronger evidence for the tile direction at h/z: tile-only
execution takes 0.494815 ms versus 0.661507 ms with neither tile bypass nor
short-row execution, a **25.20%** reduction against that padded custom fallback.
All 15 processes qualify, with bitwise frozen/candidate logits and h/z operands.
It is not a 25.20% gain against optimal dense execution or proof that training
with a new tile gate preserves quality. With tile bypass already active,
short-row execution adds approximately 1.661 microseconds here.

## Candidate next comparisons

First, distinguish gate sign from tile structure. A proposed diagnostic would
capture pre-gate a/m tensors, learned LayerNorm scale/shift, signed counts,
near-zero mass, RMS, and squared magnitude removed by each candidate gate.
Compare a-only, m-only and their joint intervention. Equal kappa and matched
zero fractions answer different questions; retain both views, plus a quality
frontier. Histogram marginals alone cannot recover joint tile occupancy.
A frozen-checkpoint post-hoc comparison measures immediate sensitivity;
it cannot establish how a model trained with the new gate would adapt.

Second, a proposed matched training comparison could keep h/z operators and
h-only OL1 fixed while varying unilateral versus bilateral a/m gates. Keeping
pressure placement and its normalization fixed avoids adding the pressure-scope
confound. h is an activation replacement in this implementation, so changing
its gate also changes the MLP nonlinearity; leave that as a separate question.
z is signed and may deserve its own bilateral comparison rather than being
assumed equivalent to h. Diagnostic and training designs remain unapproved.

Third, tile training should specify a fixed partition aligned with an intended
skippable operand tile, the activation axes, gate trigger, gradient rule,
pressure objective, pressure capture before or after the gate, and masking
cadence. The user's any-zero trigger should be evaluated as its own proposed
rule: it can erase large surviving values elsewhere in the tile. A group
magnitude criterion is a different candidate intervention. Zeros created by
the gate should not recursively trigger further tiles unless that behavior is
explicitly intended. Group pressure is a new objective; retaining OL1's update
geometry would not make it the existing orthogonal_l1 method.

Tiles spanning several tokens fit the existing uncached full-sequence workload;
the one-token decoding case requires its own definition. Zeroing activations
during training alone does not establish training acceleration: forward and
backward kernels must exploit the layout, with gate and metadata overhead
included. End-to-end timing requires a qualified efficient dense control,
the same sparse implementation with skipping disabled, and numerical checks.

The diagnostic favors resolving the a/m gate question before combining it
with tile gates, so any result can be attributed to a declared intervention.
Success would mean less quality loss at comparable sparsity, more skippable
tiles at comparable quality, and a measured net latency gain. Failure of one
axis would delimit the method rather than establish all three.
