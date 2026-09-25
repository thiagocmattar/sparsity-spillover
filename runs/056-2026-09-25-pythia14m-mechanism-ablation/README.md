# Run056: tile bypass and short-row execution at 14M

Status: design and retention approved on 25 September 2026. Implemented and
locally tested; scientific execution and RunPod launch await explicit approval.
The manuscript and supplementary archive are unchanged.

## Approved comparison

The [approved design](../../analyses/024-2026-09-17-h-only-kernel-latency/MECHANISM-ABLATION-DESIGN.md)
asks how much the two mechanisms help conditionally, jointly and alone in the
retained Pythia-14M T7/Pall kappa=0.5 checkpoint. Its weight SHA256 is
`f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.
Original random initialization, seed 1234, AdamW/OL1 history, step 712 and
1,493,172,224 training tokens are unchanged. No training occurs here.

Five modes: untouched K050; neither mechanism; tile only; short-row only;
both. Toggles affect h/z jointly in all six layers. Gates, weights, other model
paths, joint fusion, 8-row grouping, padded matrix layout and rounding remain
fixed. Short-row execution includes bias-only zero rows and scalar execution
for one/two nonzeros, subject to the existing numerical guards. Whole-group
completion belongs to short-row execution. With tile bypass off, any group
still needing matrix execution traverses every K16 step.

The resulting conditional effects need not sum to the joint saving. The
interaction is `t10 + t01 - t00 - t11`; positive means joint savings exceed the
sum of standalone savings. The padded custom dense fallback is not an optimal
dense reference, and these effects do not decompose memory and arithmetic time.

## Measurement contract

- One physical RTX 5090, BF16, batch one, length 2,048, uncached causal
  inference, all 50,304 logits per token, CUDA graphs; no concurrent GPU tests.
- Three fresh processes per mode: 15 scientific processes. The same 64 timing
  identities, seven paired passes, timing seed 2504 and runtime seed 2801.
  Synchronized host geometric means are primary; raw device timings remain.
- Full validation in every process: all 500 MiniPile documents, 338 complete
  blocks, 692,224 input tokens, 691,886 prediction tokens; 1,444-token tail
  excluded. Cache and tokenizer identities are inherited by hash from Run029.
- Eager-anchored logit atol .25/rtol .02, relative L2 <= .02 and pooled loss
  difference <= .001; additionally exact candidate/frozen graph logits and
  post-gate h/z operand bits. No tolerance relaxation after observation.
- Five full diagnostic passes (replicate 1); exact h/z operand comparisons
  in all 15 processes. Record zero/one/two-nonzero rows, guard rejections,
  original/newly empty tiles, whole-group exits, matrix/scalar work, source
  weight requests, exact/near-zero counts, RMS/L2 and weight norms.
- Kernel counters include padded work. Inherited BF16 activation-only logical
  opportunity excludes weight zeros/probability underflow and stays labeled
  as a lower bound. Original canonical logical-product evidence remains retained.

## Implementation

- `01_prepare.py` verifies and copies 1,382 frozen source/input identities;
  `build_controls.py` generates checked, minimal edits to K049 `joint.cu`.
- `site_controls.py` and `replay.py` install independent B/S switches without
  changing other K050 paths. `mechanism_reference.py` independently derives
  expected work and distinguishes signed-zero bits in exactness checks.
- `02_benchmark.py`, `qualification.py` and `diagnostics.py` retain paired
  native/candidate timings, eager qualification, a separate frozen graph, and
  full-coverage operand comparisons with instrumentation outside timing.
- `03_execute.py` runs five smokes followed by 15 randomized scientific
  processes. It requires RTX 5090 component qualification, passing smokes and
  enough time before the retrieval deadline. Failures stop progression.
- `06_cuda_checks.py` tests 12 synthetic patterns x four configurations,
  including zero/short/mixed/dense, unsafe values/weights, gate boundaries and
  one-branch-only completion. Count-on/off outputs and independent work counts
  must agree. `--development` cannot satisfy the target launch gate.
- `07_reduce.py` audits raw samples, identities, exactness, counts and device
  consistency; `13_report.py` creates the table/PDF/observation only after
  frozen timing fidelity is accepted. Non-overlapping t11/frozen process
  ranges require review; overlap is descriptive, not proof of equivalence.
- `14_compiler.py` retains JIT binary hashes and actual CUDA resource reports.
  `08_collect.py` and `09_verify_retrieval.py` archive and verify terminal files.
  Collection occurs after the worker exits so its logs cannot change in transit.

## Prelaunch verification

- Focused CPU tests: **27 passed**; independent row/tile reference, interaction
  algebra, signed zero, exactness versus numerical tolerance, input identity,
  process matrix and integer-count serialization.
- Full bootstrap suite: **242 passed**.
- Local synthetic GPU checks: **48 passed**, all outputs bitwise equal to
  frozen K049, on the RTX 5070 Ti Laptop through the configured WSL environment.
  No model checkpoint was evaluated or scientific latency measured locally.
  The compiler log and results are in `prelaunch/cuda-development.*`.
- Target RTX 5090 compilation/components, five full-model smokes and the
  scientific measurements are pending launch. Smokes use four timing inputs,
  two passes and eight correctness/diagnostic blocks; they are not paper evidence.

Reproduce CPU checks from the repository root:

```text
.venv/Scripts/python.exe -m pytest runs/056-2026-09-25-pythia14m-mechanism-ablation/test_mechanisms.py -q
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe runs/056-2026-09-25-pythia14m-mechanism-ablation/01_prepare.py verify
```

## Proposed launch envelope

One new on-demand Community RTX 5090, 32 GB, CUDA >= 12.8, any available
location, named `run056-mechanism-001`. Live catalog reads on 25 September
show Community stock LOW at **USD0.69/hour**, Secure stock NONE at USD0.99/hour.
Prefer Community; a Secure fallback is allowed in the proposal only if stock
becomes available and price remains <= USD0.99/hour. No resource is created
until launch approval. Raw catalog/inventory reads are retained in `prelaunch/`.

Use the previously verified image
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`,
30 GB container disk and 25 GB Pod volume at `/workspace`; only SSH exposed.
Pin the archived Python 3.12 / Torch 2.11.0+cu128 / Transformers 5.12.1 /
NumPy 2.5.0 environment. Install the venv and compiler cache on `/opt` to avoid
the slow mounted-filesystem installation encountered in Run037; all science,
inputs, logs, manifests and retrieval archives remain on the Pod volume.
The environment can be rebuilt after a stopped-Pod recovery.

Historical Run037 full-model processes averaged 37 seconds with roughly
3.03 GiB peak allocation. This run adds a frozen model/graph and extra exactness
checks: reserve approximately 6 GiB as an uncalibrated working estimate on
32 GB hardware. Stop if less than 2 GiB remains after graph capture. The local
synthetic allocation is not used to claim full-model resource fit.

Estimated end-to-end ETC: **35-65 minutes**, including installation, compilation,
the 15-process matrix, diagnostics and transfer. This is a historical estimate,
not calibration of the new full-model controls; refresh it after target smokes.
Maximum: **90 aggregate GPU-minutes and USD2 incremental total**, including
temporary storage and infrastructure retries. Reserve the final 10 minutes for
collection. At the current Community rate, 90 GPU-minutes costs USD1.035;
even the proposed USD0.99/hour fallback costs USD1.485. Running storage for
55 GB adds about USD0.0115 for 90 minutes at USD0.10/GB/month (720-hour month).
RunPod lists no ingress/egress fee; see [provider pricing](https://docs.runpod.io/pods/pricing).

Input transfer is approximately 94 MB, with exact bundle hashes recorded before
launch. Expected outputs are 10-30 MB uncompressed: all raw timings, per-block
checks, diagnostics, source/config/provenance, compiler reports, setup/worker
logs, reductions and any failed outcomes. The original checkpoint/cache remain
verified locally. No new checkpoint or optimizer state is produced.

## Monitoring, recovery and teardown

Execute detached with durable logs. Arm `10_deadline_guard.py` on the Pod and
in a separate hidden local Python process, each scoped to this new Pod's exact
ID/name and the same absolute deadline. Temporary credential files are consumed
and unlinked; neither keys nor guard settings enter Git. Deadline guards stop
compute and preserve the Pod volume. Resolve retrieval and deletion promptly
because stopped storage continues billing.

Monitor every 60 seconds, or at a shorter projected completion window. Report
completed processes/blocks, current validation loss, throughput and refreshed
ETC. Stop progression on numerical/exactness/count failures, CUDA errors,
nonfinite values, insufficient VRAM, stale execution progress or projected
budget overrun. Compilation progress is read from compiler logs.

After the worker is terminal, collect its artifacts, copy the archive locally,
verify archive and every-file SHA256/size, run the reduction audit, then delete
only this task's Pod and confirm absence. Failure artifacts follow the same
retention/verification procedure. No new network volume or endpoint is needed.
The pre-existing stopped Run052 Pod `qa00sm05lmrs7w` and shared 100 GB volume
`9luykg5yc3` remain untouched; their continuing storage is outside this new
incremental envelope. No endpoints were present in the inventory.
