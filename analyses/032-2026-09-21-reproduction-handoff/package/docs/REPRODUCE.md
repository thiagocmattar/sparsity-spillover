# Reproduction guide

Run commands from the repository root in the installed environment. CPU result
reconstruction uses only this release. Data preparation downloads pinned public
inputs. Training and benchmarking are separate, explicit GPU jobs.

## 1. Check the methods and retained results

```bash
python scripts/reproduce.py verify
python -m pytest -q
python scripts/reproduce.py results
python scripts/reproduce.py figures
python -m training.train --list
python -m training.train --condition 14M-T2-Ph-0.1 --smoke --output outputs/smoke-hz
```

The smoke uses a tiny synthetic CPU model and two updates, testing the actual
boundary, gates and serialization. Its losses are not paper measurements.
Read `configs/paper-grid.json` for every resolved condition. Conditions are named
by model size, topology, pressure target and threshold, matching the paper.

## 2. Rebuild the data

```bash
python scripts/prepare_data.py --splits validation
python scripts/prepare_data.py --splits train
```

The builder checks immutable Hugging Face revisions, all document/token counts,
and exact token-file hashes. Training tokens occupy about 5.97 GB; allow extra
space for the source download and checkpoints. Never substitute an arbitrary
MiniPile split. `docs/DATA.md` records identities and excluded tails.

## 3. Train one paper condition

For a new independent replication with the paper's seed and initializer:

```bash
python -m training.train --condition 14M-T2-Ph-0.1 --fresh-initialization --output outputs/14m-t2-k01
python -m training.train --condition 31M-T7-Ph-0.1 --fresh-initialization --output outputs/31m-t7-k01
python -m training.train --condition 70M-T7-Ph-0.5 --fresh-initialization --output outputs/70m-t7-k05
```

For initialization matching, replace `--fresh-initialization` with
`--initial-state /path/to/canonical-untrained.safetensors`. The parameter hash
must equal the paper cohort's recorded hash. There are no pretrained-weight
downloads, automatic hash substitutions or silent fallbacks. The CPU/GPU RNG
realization can differ; seed equality alone is insufficient. Reuse identical
untrained tensors across compared conditions, record the realized identity, and
label a changed draw as a new replication. The lean release does not supply a
public checkpoint hosting service.

The trainer records the resolved config, initial state, per-update loss,
clipping/gradient/OL1 geometry, step-one validation, final checkpoint, recovery
state and full final diagnostics. It refuses output reuse and stops on a skipped
optimizer boundary. It saves initial/final models, rather than reproducing the
historical dense checkpoint cadence. Keep stdout in a persistent log and use
your platform's detached-job mechanism for long work.

Before committing to the grid, measure one exact full-size workload, inspect
headroom and expected duration, and obtain the compute owner's approval. At
least one 80 GB-class GPU was used for the historical 14M pressure recipe;
larger models also used large-memory accelerators. No resource is provisioned
by this repository.

## 4. Evaluate and diagnose

```bash
python -m training.evaluate --checkpoint outputs/14m-t2-k01/final --output outputs/14m-t2-k01/reevaluation.json
python scripts/clip.py --checkpoint outputs/base/final --output outputs/base/clipping
```

Final evaluation reports ordinary validation loss separately from the activation
pass and eager logical-products pass. It retains site/layer exact/near-zero
counts at 0/.001/.01, RMS/L2 statistics, weight norms, six-operation integer
counts and analytic ceilings. Clip calibration uses all values from the first
ten complete source-order training blocks, independently by site/layer at
a,m,h,z. Thresholds are empirical order statistics; `abs(x) <= threshold`
is set to zero. All ten targets 0,.1,…,.9 evaluate all 338 validation blocks.

Gradient conflict and OL1 traces must be captured during training; neither a
final checkpoint nor a post-hoc activation histogram can reconstruct them.

## 5. Qualify and time final kernels

Use the Linux CUDA environment described in `environment/README.md`. Fetch
headers once, then run each measurement in three **fresh Python processes**:

```bash
python kernels/fetch_dependencies.py
python scripts/benchmark.py --checkpoint outputs/70m-t7-k05/final --output outputs/timing-r1 --replicate 1
python scripts/benchmark.py --checkpoint outputs/70m-t7-k05/final --output outputs/timing-r2 --replicate 2
python scripts/benchmark.py --checkpoint outputs/70m-t7-k05/final --output outputs/timing-r3 --replicate 3
```

The harness uses the final implementation at each size. Add `--control native-hz` to replace only h,z execution
on that checkpoint (14M/70M only). `--control hz-skips-off` is the inefficient custom fallback
ablation, not the practical native-dense control. Benchmark the Base checkpoint
in the same session to obtain the paper's across-recipe denominator. The harness
reports a same-checkpoint native ratio; do not relabel it as Base-relative speed.

Qualification uses eager BF16 reference logits on every validation block:
finite outputs, |Δlogit| ≤ .25 + .02|reference|, per-sequence relative L2 ≤ .02,
and absolute mean-loss difference ≤ .001. Both native and candidate CUDA graph
outputs are checked. Failed qualification is retained and prevents timing.

Timing includes embeddings, all blocks, gates, inspection, final normalization
and all 50,304 logits. Static layout preparation, compilation, graph capture
and equal input staging are excluded. Each process keeps 64 fixed validation
sequences × seven randomized paired passes, both host/event timings, and all
qualification rows. Pool all 1,344 host observations per implementation across
the three processes geometrically. Do not pool GPU sessions or infer a causal
skip saving from a comparison between different checkpoints.

Aggregate the three qualified processes with:

```bash
python scripts/aggregate_timings.py outputs/timing-r1 outputs/timing-r2 outputs/timing-r3 --output outputs/timing-summary.json
```

For Table 4, use the same 14M T7/Pall κ=.5 checkpoint and run three fresh
processes for each `--operation-mode full`, `without-a`, `without-m`, `without-h`,
`without-z`, `without-qk`, and `without-pv`. Also retain `frozen`, `off` and
`projection` controls. `full` switches on all six paths; `frozen` uses untouched
final 14M implementation. Add `--diagnostics` to the first replicate of each mode for full-validation
MMA and actual BF16 operand counters. Saved time is off-mode latency minus full
latency. Its min/max span is [min(off)−max(full), max(off)−min(full)] across the
three process means; it is conditional and not additive. This ablation's all-on
latency is not interchangeable with published final-kernel times from another session.

For the 14M T2/Ph ?=.1 control, `--hz-mode A`, `B`, `C`, or `D`
selects (h on, z on), (h off, z on), (h on, z off), or (h off, z off).
Only output-projection switches change; other kernels stay frozen. These modes
are mutually exclusive with other controls. The portable CLI pairs each mode
with native execution, whereas the recorded factorial experiment paired all
four modes within each process. Its historical paired effects are reconstructed
from `14m-t2-execution-controls.json`; separate CLI jobs are new measurements,
not an exact replay of that pairing. No 31M skip-control result is claimed.

## Read the output before extending the experiment

One seed and the fixed 1.493B-token budget limit conclusions. 410M is a stress
test with a different LR and optimization regime; no final 410M specialized
kernel is claimed. Changes to gates, pressure sites, precision, validation,
token order or runtime need a new named comparison and new qualification.
