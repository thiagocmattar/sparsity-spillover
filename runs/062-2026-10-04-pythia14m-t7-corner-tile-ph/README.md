# Run062: 14M T7/Ph with causal corner tiles

Status: design confirmed by the user ("Confirmed. Go on"). Implementation and
prelaunch checks are recorded. Launch confirmed by the user ("yes") within the
USD30 packet; provisioning and execution evidence will be appended below.

## Question and matched comparison

Does extending the corner-tile gate to all seven T7 ports improve structured
sparsity and the quality-latency trade-off relative to the orange scalar T7/Ph
series? Train only kappa 0 and .1. Compare to the retained Run032 scalar controls
and the previous Run060 T2/Ph tiles. The pressure target remains h only.

All six layers gate a, m, h, q_post, k_post, v and z. Partition each sequence
independently into fixed 8-token by 16-feature tiles; q/k/v partition each head
independently. Token/feature starts are multiples of 8/16. Inspect only the first
feature of the first token of each tile:

```text
a/m/h/z: drop if raw_corner <= kappa
q_post/k_post/v: drop if abs(raw_corner) <= kappa
output[whole_tile] = 0 if drop else original input[whole_tile]
```

Equality rejects in the input dtype. A surviving tile retains every interior
value, including negatives and values below threshold. Decisions do not cascade.
The detached mask gives identity gradient on admitted tiles and zero gradient on
rejected tiles. q/k are gated after RoPE; h replaces GELU; z is concatenated PV
before W_o. a/m are distinct learned affine LayerNorm outputs. Gates at time t0
depend only on its prefix and affect t>=t0, preserving causality. Cached decoding
is rejected because it requires additional persistent mask state.

This preserves T7's signed-versus-magnitude convention but changes the complete
operator, including interior value preservation and equality. It does not isolate
geometry alone. At kappa 0 a magnitude tile drops only for an exact-zero corner;
the grouped rule need not be an identity on such a tile.

## Scientific contract

- Pythia-14M architecture/tokenizer revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`: six layers, D128, FFN512,
  four heads of width32. Random small_init/wang_init, with the verified untrained
  parameter replay (`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`).
- MiniPile revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`, all 1M train
  documents, EOS per document, canonical cache hashes. Model/data seeds1234;
  ordered-block schedule SHA256
  `f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.
- 712 updates, global batch1024, T2048, MB32/GAS32: 1,493,172,224 input tokens
  per condition. 729,088 scheduled blocks, including714 wraps beyond the cache.
- Dynamic FP16, FP32 parameters and moments, flash SDPA, zero dropout. Fused
  AdamW betas .9/.95, eps1e-8, decay .1 excluding bias/norm; task clipping1;
  peak/min LR .001/.0001, 1% warmup, GPT-NeoX-v1 pre-step cosine mapping.
- `orthogonal_l1` on post-gate h only: mean of six tensor mean absolute values,
  lambda1, task-relative trust budget1, eps1e-12, task-only AdamW moments.
- Pinned Python3.12, Torch2.11.0+cu128, Transformers5.12.1; container digest in
  config. This reuses the tested recipe mapping, not exact DeepSpeed reproduction.
- Every validation/diagnostic pass: all500 documents,338 complete2048-token
  blocks,692224 inputs,691886 predictions; report excluded tail1444.
- Full loss at 0/1/32/128/256/512/712; activation/tile diagnostics at
  0/32/128/712. Model checkpoints at 0/1/2/4/8/16/32/64/128/256/512/712;
  optimizer/scaler/RNG recovery at256/512/712. Retain all final checkpoints.

## Diagnostics and inference follow-up

The user confirmed the full proposed post-hoc inventory. Collect exact/near-zero
integer counts at0/.001/.01 and the trained threshold, pre/post RMS/L2/moments,
removed energy, per-layer and pooled row/empty-tile counts at1x16/8x16/16x16/32x16,
corner certificates, block-position counters, LN affine values, weight norms,
actual six-operation logical products and the declared T7 reach ceiling. Capture
gradient conflict and OL1 geometry at every training boundary; no clipping sweep.
Post-gate h pressure cannot propagate an input gradient through a rejected tile.

`latency/` contains the approved follow-up. Stage the final tiles only after the
training verifier passes, then measure them alongside the retained scalar T7/Ph
checkpoints c36/c39 on one physical RTX5090. BF16 B1/T2048, uncached, full50304
logits,64 fixed validation inputs x7 interleaved passes x3 fresh processes per
threshold:1344 paired samples per comparison. Runtime/timing seeds2801/2504.
Native, sparse and no-h/z-skip graphs each qualify against their own eager model
over all338 blocks (logit atol.25/rtol.02, relativeL2<=.02, pooled loss delta<=.001).
Keep original canonical FP16 quality separate from BF16 qualification loss.

The non-target laptop development smoke completed all four models. Native graphs
passed; optimized c39 and untrained tile kappa 0 failed the declared logit bounds.
The operand trace shows unsplit native laptop attention versus the preserved
two-split RTX5090 kernel. This check supplies no accepted latency and cannot
qualify final trained models. Target-GPU smokes and full validation remain
mandatory. See `latency/prelaunch/native-cuda-smoke-003.json` and the dispatch
trace; numerical tolerances were not relaxed.

The h/z consumer is the unchanged Run061 CUDA source. K050 rounded norms and
K019 unthresholded RoPE outputs feed one exact a/m gate pass and one headwise
q/k/v gate pass. Scalar and corner modes share the same compiled gate code.
K042 input projections and K035 attention remain unchanged. These recurring
passes are timed. The no-skip switch disables only h/z tile bypass and short-row
substitution; other sparse consumers remain fixed. Historical orange timings
used different gate fusion and sessions, so retain them as descriptive
coordinates and use the new paired controls for controlled runtime claims.

Support: useful loss improvement, surviving activity and more empty tiles, with
an explicit quality/latency trade-off. Collapse/stalled learning, nonfinite or
skipped updates, large loss penalties or no measured speed benefit weaken the
hypothesis at this budget. No numerical acceptable-quality margin was specified.
One training seed, two thresholds and one inference GPU do not establish
portability, equal-quality advantage, cached-decoding speed or representational
redistribution. Logical opportunity and MMA bypass do not measure DRAM traffic.

The manuscript connection is `draft/05-experimental-study.tex`, "Targeted and
broad thresholding across model sizes", and its orange T7/Ph series. The grouped
operator extends the elementwise definitions in `draft/04-methodology.tex`;
operational Ph stays unchanged. Comparison figures belong in Analysis041.
No manuscript edit or finding promotion is included.

## Implementation and provenance

The training adapter derives from Run060, with original source hashes in
`prelaunch/derived-sources.json`; historical files are unchanged. `tile_gate.py`
implements signed/magnitude corners for 3D/4D ports. `diagnostics.py` captures
all42 gate inputs and pools within-head counts; `03_verify.py` reconciles
90 activation rows,42 corner/energy/occupancy rows, all boundaries, checkpoints
and transferred hashes. `latency/provenance/` identifies frozen consumers and
the retained scalar controls. Development tile models are explicitly untrained
and are excluded from scientific benchmark inputs.

Prelaunch results and execution definition are in [LAUNCH.md](LAUNCH.md) and
`prelaunch/checks.json`. Mandatory exact-condition H100 preflights discard their
state, verify initial/data/runtime identity and memory headroom, and restart
scientific training from the approved step-zero bytes. The RunPod router,
runpod-mcp and runpodctl skills were read. Current schemas and CLI2.14 help were
checked; no legacy deadline-create flag is assumed.

Regenerate checks from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/test_run062.py -q --basetemp=.pytest_tmp/run062-focused
.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_tmp/run062-bootstrap
.venv/Scripts/python.exe -X utf8 runs/062-2026-10-04-pythia14m-t7-corner-tile-ph/09_local_smoke.py
```

## Approved execution: attempt001

The user's explicit launch approval followed design approval and the separate
launch packet. Pod `iwoydp694czykb` was created at 2026-10-04 17:58:24 UTC:
two H100 SXM GPUs in EUR-NO-2, actual host price USD6.98/hour. Its two-hour
deadline is 19:58:24 UTC; detached workload and independent workstation provider
stop guards were armed before setup. The USD30 combined training/inference
ceiling remains in force. See `prelaunch/lease-training-001.json`.

The executed checkout is `03ff9227e3700bf93802d6a38533d5cc815bef66`, uploaded
as source bundle002 and checked remotely by byte count and SHA256. Both canonical
token caches were rebuilt and their hashes verified. The real exact-condition
preflights precede fresh scientific training; no preflight state becomes a
scientific checkpoint.

After training, `17_scalar_structure.py` will evaluate the two retained scalar
controls with native FP16 autocast and complete validation, without optimization
or clipping. This adds comparable seven-site zero and tile occupancy counts.
Its explicit script and input hashes distinguish this post-hoc measurement from
the already running, unchanged training checkout. All agreed training and
post-hoc artifacts must be sealed and verified locally before Pod deletion.

Analysis041 scripts03/04 prepare the seven-site table, learning/occupancy PDFs
and the orange-frontier comparison. New optimized latency coordinates require
successful RTX5090 qualification; preparation of plotting code is not a result.

## Verified training and transfer

Both exact-condition H100 preflights passed: post-warmup median3.5366/3.5363s,
61,501,079,552 reserved of85,017,493,504 bytes (72.3%). Both fresh scientific
conditions completed all712 updates without overflows or skipped boundaries.
Remote and local scientific verification passed. The1,877,912,761-byte archive
and all148 members were checked; all24 model checkpoints and six recovery states
are retained locally. The owned H100 Pod was deleted and its absence confirmed.

| kappa | Scalar canonical FP16 loss | Tile canonical FP16 loss | Tile minus scalar | Perplexity change |
|---:|---:|---:|---:|---:|
| 0 | 5.219827 | 5.721071 | +0.501244 | +65.077% |
| .1 | 5.237395 | 5.835102 | +0.597707 | +81.795% |

Both conditions learn with finite activations and exact corner certificates;
the fixed-budget quality cost is substantial. The full seven-site table is in
Analysis041 `results/t7-comparison.md`. Scalar occupancy comes from the separate
complete FP16 replay; its loss differs from the original canonical scalar loss
by less than.00002. The local verifier's identical result was saved separately
and original transferred LF bytes restored after its Windows CRLF rewrite, so
the transfer inventory remains exact. See `prelaunch/local-verification-training-001.json`.

The approved four-checkpoint RTX5090 follow-up is now allocated as Pod
`7ugmnw8p4ksb74`, EU-RO-1, USD.99/hour, provider start19:06:03 UTC and one-hour
deadline20:06:03 UTC. Its independent provider stop guard is armed. Optimized
latency remains pending numerical qualification.
