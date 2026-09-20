# Run043: Pythia-70M h/z gates and h-only OL1

User authorized promotion and parallel RunPod execution on 20 September 2026,
requesting the fastest completion. This carries forward Run041's four thresholds
0/.01/.05/.1, one-sided gates at h/z only, h-only OL1 lambda=1, trust budget=1,
no post-hoc clipping, complete diagnostics and final specialized-kernel latency.
The new question is whether the low-loss h/z regime transfers to the established
70M recipe. Comparisons with retained 70M endpoints will be descriptive, one seed;
improved loss/sparsity/runtime operating points support transfer, deterioration
or failed kernel qualification limits it. No manuscript edits or finding promotion.

## Scientific contract

Match Run034/Run018's six-layer, width512, FFN2048, eight-head random Pythia-70M
architecture revision e93a9faa9c77e5d09219f6c868bfc7a1bd65593c. Load only the
retained canonical random initialization and RNG, never released/trained weights.
Parameter SHA e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062;
schedule SHA d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e.
Model/data seed1234; pinned MiniPile, EOS/document; 712 updates, MB4 x GAS256,
global1024, sequence2048, 1,493,172,224 input tokens per condition. Fused AdamW
(.9,.95), eps1e-8, weight decay.1 except bias/LN; task-gradient clipping1;
peak/minimum LR .001/.0001, 1% warmup, pre-step cosine. FP32 parameters/moments,
dynamic FP16, flash SDPA, zero dropout, no activation checkpointing.

h replaces GELU and z is immediately before W_o. Values x>=kappa survive,
otherwise become zero; the mask is detached. At kappa0 both gates are ReLU.
Pressure averages six post-gate h tensor means, with no other pressure site.
All other scientific sections match Run034. HZ's analytic reachable count is
16,106,127,360 / 104,293,466,112 products per B1,T2048 full forward; observed
R_model includes natural zero operands and is not a speedup.

Full validation after step1 and the reloaded final checkpoint uses all500 source
documents,338 complete blocks,692224 input tokens,1444 excluded tail tokens.
Final activation and logical passes have identical coverage. Retain exact/near
zero counts at0/.001/.01, RMS/L2/finite statistics at a,m,h,q_post,k_post,v,z and
attention_output; all named weight norms; integer logical counts and analytic
ceiling; every boundary's gradient conflict, OL1 geometry, loss scale and clipping.
Models:0,1,2,4,8,16,32,64,128,256,512,712. Full optimizer/scaler/RNG:256,512,712.
All48 models and12 recovery states will be copied locally (about20.3GB total).

## Execution envelope

One Secure four-H200 Pod preferred, four independent GPU workers, no DDP.
Live quote USD4.59/GPU-hour,18.36/hour total; H100 SXM fallback3.49/GPU-hour.
Four-hour Pod stop deadline; USD80 total incremental cap including later latency,
temporary disk and transfer. Prior comparable70M throughput varies139k-305k
tokens/s; expected completion roughly2-3h, refined by exact preflight. The user
explicitly selects parallel cloud execution; no local GPU training is scheduled.
40GB container,100GB isolated persistent /workspace, pinned Run034 image.
The separate Run042 Pod and existing network volume are not used or modified.

Detached pipeline installs the pinned runtime and rebuilds the token cache with
complete SHA256 verification while canonical random tensors upload. Each GPU
preflights six real boundaries (one warmup), full validation and checkpoint save;
the highest threshold also calibrates full activation/logical passes. Training
starts fresh only when all pass and ETC fits the deadline. Remote process-group
guard is credential-free; a separate hidden local guard performs provider stop.
No account credential leaves the workstation. Deadline stop preserves storage.

Read-only monitoring every5min, diagnostic polling60s, shortened near completion.
Report progress, loss, throughput and refreshed ETC. Warn on nonfinite/skipped
updates, hash/capture/runtime drift, stale10min events, <10% VRAM headroom,
<5GB disk, or projected deadline/cost overrun. Hash-verify archives and every
required member locally before deleting a Pod. Final audit excludes unrelated
resources from teardown and cost claims.

Final latency will use the frozen qualified Run035 k050-70m-v2 implementation,
adapted only to accept HZ topology: RTX5090, BF16, B1,T2048, uncached full50304
logits, native SDPA graph denominator, native eager correctness anchor. Three
fresh processes per model,64inputs x7paired passes,1344pairs each; full338-block
qualification at atol.25+rtol.02,relativeL2<=.02,absolute pooled loss delta<=.001.
No new kernel search. Reserve up to90min at the current0.99/hour quote, within
the globalUSD80 cap. Clipping remains disabled.

## Verification and execution record

Prelaunch verification: four focused tests passed in12.71s, including a real70M
CPU canonical-weight load, exact initial hash, all12 h/z gate captures, six h
pressure tensors, threshold equality/gradient behavior, checkpoint roundtrip,
matched70M scientific sections/data schedule and exact analytic ceiling. Full
bootstrap suite242 passed in8.89s. Commands used `.venv/Scripts/python.exe -m
pytest -q runs/043-2026-09-20-pythia70m-hz-h-only-ol1/test_run043.py` and the same
command with `tests`, each using a fresh run-specific `--basetemp`.
These CPU tests do not qualify CUDA; exact four-GPU preflight remains mandatory.
Local free disk is817.8GiB, sufficient for all20.3GB of model/recovery outputs,
compressed archives and latency inputs. The validated configuration is ready
for the already-authorized launch. Calibration, allocations and receipts follow.
Executed scientific source remains frozen; infrastructure retries get separate
attempt records.

The allocated training Pod is `25a3geba9jatml`, four H200s in EUR-IS-4 at
USD18.36/hour, created14:46:01UTC on20September. Its provider-stop deadline is
18:46:01UTC. The isolated deployment commit is
`1685e87948cc57314ba3252e6b1dbbb4092e7cc0`, based on prelaunch commit
`02c9b1eb92af2cb7bd524c71bdeb28d744f8258e`; the source bundle and both canonical
initialization files passed remote SHA256 verification. See the sanitized
`prelaunch/lease-training-001.json`, source receipt and initialization upload
receipt. Runtime setup overlaps the initialization upload; execution is detached.

`19_prepare_latency_sources.py` freezes1374 historical source files and the
Run035 70M port. All CUDA/header/operator sources are byte-identical to its
qualified v2. The composition file redirects only the adapter import to the HZ
allowlist bridge. Three local latency-contract tests pass, covering the12-process
matrix, unchanged tolerances, exact kernel source identities and the bridge.
Final checkpoint inputs will be attached after training verification.
`20_retrieve_endpoints.py` copies and verifies the four terminal models first,
allowing latency execution to overlap the complete training archive retrieval.
All model and recovery checkpoints still require full local hash verification
before training Pod teardown.

All four original GPU preflights passed. The automatic25% timing margin plus
20-minute reserve exceeded the remaining lease by several minutes, so the
initial infrastructure pipeline exited before any scientific training. Its
logs and exit record remain intact. An additional six-boundary calibration
with one CPU thread per worker also passed in separate
`prelaunch/thread-calibration-002/` records; it did not improve the slowest
worker. The original eight-thread setting was retained.

Infrastructure continuation003 starts the same four unchanged scientific
workers under the original18:46UTC stop deadline. Its reserve uses measured
validation/diagnostic/checkpoint costs, a15% training-time margin and20minutes
for archiving/transfer. The resulting bound is12,434.9s versus12,744.7s remaining;
the unpadded training estimate is9,751.3s. See
`prelaunch/training-launch-003.json` and `prelaunch/continue-training-003.sh`.
The detached controller and its process-group guard were updated together.

The latency Pod `0pdyeln5tfjzt4` was allocated at17:41:45UTC: one RTX5090 in
EUR-NO-1 atUSD0.99/hour,40GB container and40GB persistent workspace, with the
same pinned image. Its provider-stop deadline is19:11:45UTC. The previous
Run042 Pod was already absent at this discovery; no unrelated resource was
modified. `21_prepare_latency_runtime.py` starts detached runtime installation
while the training workers finish, without uploading scientific inputs yet.
The final input archive is still hash-verified before any benchmark executes.
