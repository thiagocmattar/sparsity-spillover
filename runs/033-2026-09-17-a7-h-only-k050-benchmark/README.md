# Run 033 — final K050 on the five new A7 h-only checkpoints

## Approved design and launch status

The user explicitly narrowed the scope on 17 September 2026:
"benchmark only the new runs ... just the final kernel with the new models."
This supersedes the proposed 40-checkpoint refresh. This run evaluates **only
the five Run 032 final checkpoints** with final K050 and its paired native
reference, which supplies the speedup denominator. Existing A4 h-only and
other historical results are reused. No old model, A0 timing, skip-control
ablation, optimizer update, or kernel search is included.

Implementation and local checks are complete. **Cloud launch is pending the
explicit launch confirmation required by AGENTS.md.** The 70M proposal remains
frozen in Analysis 023. No existing Run 028/029/032 cloud budget is reused.

## Scientific question and matched contract

Measure final-K050 full-model latency and native-relative speedup at the five
A7+OL1(h) endpoints (kappa = 0, 0.01, 0.05, 0.1, 0.5), and add them to the
existing model-wide-sparsity/runtime figures. These are retained randomly
initialized seed-1234 Pythia-14M models after 712 updates and 1,493,172,224
training tokens. Their weights, gates, pressure identities, initialization and
realized training order remain those verified in Run 032. Gate sites are
a,m,h,z (one-sided) and q_post,k_post,v (symmetric); trained OL1 targets h only,
lambda = 1, b = 1. Original final recovery checkpoints remain retained locally.

The measurement is Run 029's **unchanged** `02_benchmark.py`, `replay.py`, and
`io_utils.py`, with the same frozen source archive and final K050 settings
(`round_p=False`, `shortcut=False`). Each new model is paired with native
PyTorch SDPA CUDA-graph execution in the same fresh process. Unmodified native
eager execution anchors correctness; it is not the speedup denominator.

Use one RTX 5090, BF16, batch one, T=2048, uncached causal inference and full
50,304-vocabulary logits. Runtime pins are Python 3.12, Torch 2.11.0+cu128,
Transformers 5.12.1 and NumPy 2.5.0, plus Run 029's retained dependency lock.
Timing seed 2504 and runtime seed 2801 are unchanged. Each model has 64 fixed
validation inputs, seven paired timing passes, and three fresh processes:
15 scientific processes and 1,344 timing pairs per checkpoint. Pool paired
ratios geometrically, not ratios of independently selected medians.
Graph setup/compilation/equal input staging are recorded separately;
recurring packing, zero detection, and full logits remain timed.

Every process qualifies all 338 complete blocks from all 500 validation
documents (692,224 input tokens and 691,886 prediction tokens); the
1,444-token tail is excluded. Preserve logit atol=0.25, rtol=0.02,
relative-L2<=0.02 and absolute loss difference<=0.001. Failed qualification
remains visible and cannot be relabeled as a speedup result.

Qualified gains support a runtime benefit in this workload; slowdowns and
failures weaken it. Canonical FP16 S_model uses retained pooled integer
logical-product counts. BF16 timing and BF16 executed-work diagnostics remain
separate measurements. Scalar sparsity is not itself a runtime or causal claim.

## Comparability and figure update

The GPU model, scientific driver, kernel, inputs, precision and protocol match
Run 029. Paired native-relative ratios support comparison across the two
sessions. A different physical GPU/host can still shift absolute latency;
retain GPU UUID, environment and session identity and disclose that difference.
Rebenchmarking the historical cohort is not required or authorized.

Add A7+OL1(h) to full-model speedup and native/K050 absolute-latency displays,
and include retained A4+OL1(h) results. A historical common-A0 ratio, if shown,
must explicitly identify its old-session numerator; it is not a newly matched
A0 timing. New projection/attention-ablation points are unavailable because
the user requested only final K050. Preserve existing ablation observations.
The manuscript connection is Section 4.6, `kernel-autoresearch.tex`; do not
silently change its speedup reference or mix loss conventions. New figure
versions, source reductions, complete tables and observation captions will
retain both sessions' provenance.

## Retained diagnostics and artifacts

Once per checkpoint, the existing full-validation K050 diagnostic collects
exact/near-zero counts at 0/0.001/0.01, RMS/L2, named weight norms, logical and
executed-skip counters, and row/fragment occupancy histograms. All three
replicates retain raw timing pairs, input indices, complete numerical checks,
losses, setup time, peak memory, source/runtime identities and failure examples.
Training-time gradient/OL1 geometry remains in Run 032; inference cannot
reconstruct it. No additional clipping sweep is included.

Upload only the five inference checkpoint files, frozen sources and full
validation cache. Optimizer/RNG recovery and other training checkpoints stay
verified locally. Return all result/quality/timing/diagnostic JSON, manifests,
logs, environment and transfer inventories, even if a qualification fails.
Verify every returned byte count and SHA-256 before normal Pod deletion.

The prepared `bundles/input-001.tar` contains 318,730,240 bytes; its SHA-256
is `d80553fd2e6302fa7284077be89f38e9c87fa258d6da981f6d6adbca2b60cec7`.
Its receipt and complete member inventory are retained beside the ignored tar.
The local-only `07_local_stop_guard.ps1` stays on the controlling machine.

## Concrete proposed execution

- One Community RTX 5090, 32 GB, currently quoted USD0.69/hour; Secure fallback
  at USD0.99/hour only within the same proposed ceiling. Live stock is low.
- Name `run033-k050-new-001`; image
  `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`;
  CUDA host >=12.8, SSH only, 20 GB container disk and 25 GB `/workspace` Pod
  volume. Leave the existing network volume untouched.
- Initial total ETC **20–45 minutes**, including setup, transfer, cold
  compilation, two endpoint smokes, 15 scientific processes, retrieval and
  verification. Warm scientific work is estimated at 4–6 minutes from the
  historical complete-process calibration; this must be refreshed on-Pod.
- Expected cost roughly **USD0.25–0.80**. Proposed maximum **90 cumulative
  GPU-minutes / USD2 total**, including setup, retries, disk and retrieval.
  No automatic extension. A deadline stop preserves storage for recovery;
  any residual storage charge must be reported until verified deletion.
- Arm both an on-Pod identity-scoped API stop guard and an independent hidden
  local guard against the same creation-time deadline before setup. The
  on-Pod guard consumes and unlinks its temporary credential file. No secret
  is logged, bundled, or committed.
- Setup and controller are detached and log under persistent `/workspace`.
  Monitor each 60 seconds or near predicted completion. Report completed
  processes, loss/error, inputs/sec or process throughput, memory and ETC.
  Stop/investigate reference failure, nonfinite outputs, CUDA failure, stale
  progress beyond two expected process durations, low disk or deadline risk.
- Normal closeout: transfer, inventory/hash checks, local reduction checks,
  delete the scoped Pod, then confirm no task Pod remains. No weights or
  datasets enter Git.

## Local verification

The focused tests passed 4/4; the full bootstrap plus this contract passed
246/246. All 1,410 frozen source/input identities verified. The tests check
five-new-model-only scope, the complete kappa grid, unchanged numerical/timing
protocol, byte-identical driver/replay, original checkpoint bytes, validation
coverage, pooled sparsity counts, K050 settings and 15 unique process jobs.
These CPU checks do not establish CUDA numerical correctness or latency.
Two non-evidence remote endpoint smokes (kappa 0 and 0.5; four timing inputs,
two passes and eight validation blocks) precede complete measurements.

Preparation handled Windows long paths for frozen CUDA headers and reused
Run 029's inline validation metadata correctly. These were packaging fixes,
not changes to the scientific source, checkpoints or measurement protocol.

## Reproduction

From the repository root, prepare and verify with `01_prepare.py prepare`
and `01_prepare.py verify`; create an immutable tagged transfer tar with
`01_prepare.py bundle --tag <new-tag>`. Run `04_setup.sh` remotely, followed
by `05_execute.sh smoke <deadline-epoch>` and `05_execute.sh scientific
<deadline-epoch>`. Preparation refuses to overwrite an executed attempt tree;
process attempts have unique names and cannot silently overwrite earlier data.
