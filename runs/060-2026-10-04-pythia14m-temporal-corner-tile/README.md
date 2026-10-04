# Run060 — 14M T2/Ph with causal corner tile gates

Status: completed and verified locally. Both 712-update conditions finished
without overflow or skipped updates. All 144 transferred files, including 24
model checkpoints and six recovery states, passed byte/SHA256 checks. Tables,
three PDF figures and observations are complete in Analysis041. The owned Pod
was terminated and provider inventory confirms its absence.

## Question and comparison

Can the model learn useful representations when h/z share a zero mask over
eight consecutive tokens and16 features, with tolerable validation-quality loss?
The user chose this strong temporal restriction and explicitly refined its
trigger to the first feature of the first token, position1,1 of each tile.

Train two random-initialized14M conditions, kappa 0 and.1, with T2 gates at h/z
and Ph pressure at h. Compare final full-validation FP16 results with the retained
elementwise T2/Ph endpoints from [Run058](../058-2026-10-03-pythia14m-t2-p2-versus-ph/README.md).
Pressure and training recipe are matched; the gate operator changes. No a/m
gate, bilateral gate, tile pressure, clipping sweep or inference timing is added.

## Exact gate and backward

Partition each sequence independently into fixed, nonoverlapping8x16 tiles:
token starts0,8,16,... and feature starts0,16,32,... . For raw gate input X:

```text
drop = X[batch, token_start, feature_start] <= kappa
output[whole_tile] = 0 if drop else X[whole_tile]
```

Only the corner is inspected. Equality rejects in the input comparison dtype;
explicit zero rejection is redundant because kappa is nonnegative. A surviving
tile preserves every original value, including negative/subthreshold noncorner
features on the first row. There is no additional elementwise gate. Newly written
zeros do not cascade and tiles never cross sequences or batch entries.

For finite inputs, post-gate corner==0 iff the whole tile is empty. Corner>kappa
certifies an active tile. Diagnostics verify both implications on full validation.
The decision at token t0 uses only its prefix and affects t>=t0, so it is causal.
Partial final token blocks are supported for prefix tests; scientific sequences
are exactly2048 tokens. Cached decoding is explicitly rejected: a later streaming
implementation needs per-layer/site mask state and absolute token alignment.

The detached mask gives identity input gradient on surviving tiles and zero
gradient on rejected tiles. No STE/soft gate is added. Post-gate h pressure cannot
provide an input gradient through a rejected tile. Recovery is an empirical
question; capacity alone does not guarantee it. A tile mask restricts branch
features, not the complete residual token or an entire attention head.

## Matched scientific contract

- Pythia14M pinned architecture/tokenizer revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`, six layers, h512/z128.
  Random small_init/wang_init; replay the verified untrained step-zero bytes.
  Initial parameter SHA256:
  `ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`.
- MiniPile revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`, all1M train
  documents, EOS appended, source-order tokenization and hash-locked caches.
  Model/data seeds1234. Schedule SHA256:
  `f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.
- 712 updates, global batch1024, T2048, MB32/GAS32,1,493,172,224 input tokens
  per condition.729,088 scheduled blocks,714 wraps beyond the728,374-block cache.
- DynamicFP16 with FP32 weights/moments; AdamW betas.9/.95, eps1e-8, weight
  decay.1 excluding bias/norm; task-gradient clip1; peak LR.001/minimum.0001,
  warmup.01, existing GPT-NeoX pre-step scheduler mapping, zero dropout.
- h gate replaces GELU on W1 output before W2. z gate acts on concatenated PV
  before W_o, in all six layers. a/m remain ungated learned affine LayerNorm ports.
- Ph: `orthogonal_l1` on post-gate h only; mean of six tensor mean absolute
  values, lambda1, relative trust budget1, eps1e-12, task-only AdamW moments.
- Python3.12, PyTorch2.11.0+cu128, Transformers5.12.1; pinned container digest
  and full package versions in config/setup. This is the existing recipe mapping,
  not a claim of bitwise GPT-NeoX/DeepSpeed framework reproduction.
- Every validation/diagnostic pass covers all 500 documents, all338 complete
  blocks:692,224 input tokens,691,886 predictions, excluded tail1,444 tokens.
- Model checkpoints0,1,2,4,8,16,32,64,128,256,512,712; complete optimizer,
  scaler and RNG recovery states at256,512,712; retain the final checkpoint.

## Measurements and interpretation

Full loss validation at 0,1,32,128,256,512,712; activation/tile diagnostics at
0,32,128,712. Compare loss/perplexity and put exact-zero and empty 8x16-tile
percentages side by side at a,m,h,z. Retain per-layer and pooled integer counts,
additional1x16/16x16/32x16 occupancy, row nonzero histograms, near-zero mass at
0/.001/.01 and the trained threshold, pre/post RMS/L2/moments and removed energy.
Store corner rejection counts by layer/feature group, position0–7 zero counts,
corner-certificate checks, learned LN affine values, final weight norms and
per-boundary task/pressure/OL1 gradient geometry. Full logical-product counts and
the HZ analytic ceiling are auxiliary opportunities with explicit units.

Support for adaptation: ongoing loss improvement and useful surviving branch
activity with increased empty-tile structure and a measured quality trade-off.
Persistent collapse/stalled learning, nonfinite behavior or a large final loss
penalty refutes successful adaptation within this budget. No acceptable loss
penalty was numerically specified: report the actual difference. One seed, two
thresholds and712 updates cannot establish a general capacity or scaling claim.

The gate preserves signed noncorner values, unlike the scalar control. Results
therefore compare complete operators; they do not isolate geometry alone.
The hardware audit supports K16 alignment but guarantees no speedup. Upstream
W1/QKV/attention still execute. Runtime measurements remain deferred.

The new grouped operator differs from the elementwise threshold definition in
`manuscript/draft/04-methodology.tex`. It tests the quality/structured-zero trade-off
motivated there, while operational Ph stays unchanged. No manuscript text or
empirical claim is changed. [Analysis040](../../analyses/040-2026-10-04-tile-gate-hardware-alignment/README.md)
retains the hardware audit and its original source identities.

## Implementation and verification

`tile_gate.py` implements the corner rule and a narrow run-local adapter for
the reused model/topology/checkpoint helpers. Historical gate classes and source
files are unchanged. `training.py` adds cadence validation and diagnostics to the
Run058 trainer. `diagnostics.py` verifies the corner certificate and pools counts;
`03_verify.py` reconciles coverage, recovery states and transferred hashes.
`prelaunch/derived-sources.json` identifies the copied Run058 source files before
adaptation; it is provenance, not a claim that adapted runtime bytes are identical.

Focused tests:19 passed. Bootstrap:242 passed. Real14M CPU smoke: both conditions
passed one T16/MB1 FP32 boundary with identical initial parameter SHA and exact
h-only capture. CPU checks do not establish production CUDA memory fit or ETC.
See [the check receipt](prelaunch/checks.json) and [smoke receipt](prelaunch/local-smoke.json).

Commands:

```powershell
.venv/Scripts/python.exe -m pytest runs/060-2026-10-04-pythia14m-temporal-corner-tile/test_run060.py -q --basetemp=.pytest_tmp/run060-focused
.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_tmp/run060-bootstrap
.venv/Scripts/python.exe runs/060-2026-10-04-pythia14m-temporal-corner-tile/09_local_smoke.py
```

Both production CUDA preflights passed, their model/optimizer state was discarded,
and scientific conditions restarted from step-zero bytes. Median full updates
were 3.3822/3.3821s, with 61,467,525,120 reserved bytes out of 85,017,493,504 (72.3%).
The frozen cache rebuild reproduced both canonical hashes. Scientific runtime
identity remained unchanged throughout execution.

## Execution record

Owned Pod `b0jo43zzd0xvbg` (`run060-temporal-corner-tile-001`) started at
2026-10-04T12:23:51.572Z in US-MO-1. Two H100 SXM 80GB GPUs cost USD6.98/hour
in total. Actual `nvidia-smi` reports 81559MiB per GPU, driver580.159.04;
the provider's `gpu.memory=503` field in the raw lease is host memory, not VRAM.
The two-hour provider-stop deadline was 2026-10-04T14:23:51.572Z. Both the remote
workload guard and hidden workstation provider-stop guard were armed at launch;
the local guard was cancelled after verified teardown.

Deployment uses commit `92769d86c21064e5486fa9c3ea2ee7a0fe32d80f` and the
pinned container digest. Upload/source and lease receipts are under `prelaunch/`.
Existing foreign Pods and the shared network volume remain untouched.
Cross-run analysis belongs to
[Analysis041](../../analyses/041-2026-10-04-corner-tile-versus-elementwise/README.md).

## Verified result and closeout

| kappa | Scalar T2/Ph loss (Run058) | Corner tile T2/Ph loss | Change | Perplexity change | H empty 8x16 | Z empty 8x16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.127033 | 5.553770 | +.426737 | +53.225% | 57.5056% | 44.9262% |
| .1 | 5.152346 | 5.596746 | +.444400 | +55.955% | 84.3082% | 70.9567% |

The models learn (initial losses 11.0252/11.0267) and every final h/z layer retains
activity. The fixed-budget comparison shows substantially more fully empty tiles
and a substantial quality penalty, alongside fewer element zeros than scalar
gates. It supports viability of learning with this restriction, but does not
support preservation of quality in this recipe or establish redistribution.
No latency, tile pressure, new model size or manuscript change was added.

See the [full a/m/h/z comparison](../../analyses/041-2026-10-04-corner-tile-versus-elementwise/results/comparison.md)
and [figure observations](../../analyses/041-2026-10-04-corner-tile-versus-elementwise/observations/INDEX.md).
Attempts are `001-20261004-125905-e5a32225` (kappa 0) and
`002-20261004-125906-45dd6a88` (.1). The terminal archive has 1878058178 bytes,
SHA256 `74d8b68bfa97b76f2c70bb89239f7a641e6e272d80d5babc619a2d2c4c883f6b`;
all 144 members passed local verification, followed by `03_verify.py`.

Termination requested at 2026-10-04T13:50:54Z; absence confirmed at 13:51:40Z.
Estimated GPU cost is USD10.126-10.215 from provider-start/teardown bounds, with
an estimated storage allowance below USD.10, within the approved USD30 ceiling.
Posted billing is partial, not the final invoice. No owned billable resource
remains; the deadline guard was cancelled after deletion. Existing Run052's
EXITED Pod and shared 100GB volume remain untouched. See
[compute closeout](results/compute-closeout.json).

Metadata note: the immutable random replay's historical `reason` string mentions
an earlier T4 comparison. The replay is the same untrained parameter bytes;
actual resolved configs, topology, pressure captures and manifests verify T2/Ph
at h/z for this run. The historical string was retained, rather than rewriting
hashed input provenance.
