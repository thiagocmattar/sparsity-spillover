# Run060 — 14M T2/Ph with causal corner tile gates

Status: approved design implemented; local prelaunch checks passed. Scientific
training and cloud provisioning have not started. The proposed billable envelope
is in [LAUNCH.md](LAUNCH.md); launch confirmation remains pending.

## Question and comparison

Can the model learn useful representations when h/z share a zero mask over
eight consecutive tokens and16 features, with tolerable validation-quality loss?
The user chose this strong temporal restriction and explicitly refined its
trigger to the first feature of the first token, position1,1 of each tile.

Train two random-initialized14M conditions, kappa0 and.1, with T2 gates at h/z
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
- Every validation/diagnostic pass covers all500 documents, all338 complete
  blocks:692,224 input tokens,691,886 predictions, excluded tail1,444 tokens.
- Model checkpoints0,1,2,4,8,16,32,64,128,256,512,712; complete optimizer,
  scaler and RNG recovery states at256,512,712; retain the final checkpoint.

## Measurements and interpretation

Full loss validation at0,1,32,128,256,512,712; activation/tile diagnostics at
0,32,128,712. Compare loss/perplexity and put exact-zero and empty8x16-tile
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

The approved launch would run two production CUDA preflights first, then discard
their model/optimizer state and restart both scientific conditions from step-zero
bytes. Local artifacts are retained; scientific results/figures remain pending.
