# Run 058: 14M T2/P2 versus T2/Ph

Status: all four conditions completed and verified locally. Tables and two PDF
figures are retained; the owned Pod is deleted.

On 3 October 2026 the user requested T2 (one-sided gates at h,z only), P2
(pressure at h,z), compared with Ph, and explicitly authorized RunPod execution
following Run057 principles: “Beleza, pode rodar na run. Pode seguir nos mesmos
princípios da run anterior. Vai lá.” This authorizes design, implementation,
launch and verification without another confirmation.

Question: does expanding OL1 pressure from h to h,z improve the quality and
structured activation sparsity of the same T2 gates? Fresh T2/Ph and T2/P2 are
trained at kappa=0 and .1 in four concurrent independent workers. The only
within-threshold scientific change is the pressure target set. Neither a nor m
has a gate or pressure. h replaces GELU, z is concatenated PV before W_o;
equality survives the detached one-sided comparison in all six layers.

Match Run057 random step-zero bytes, pinned architecture/tokenizer and MiniPile
revisions, seed1234, exact realized schedule, 712 updates, global batch1024,
MB32/GAS32, T2048, dynamic FP16, FP32 parameters/moments, AdamW .9/.95,
eps1e-8, weight decay.1 excluding bias/norm, task clipping1, peak LR.001,
minimum.0001, warmup.01 and the pinned GPT-NeoX pre-step schedule mapping.
OL1 weight1, relative trust budget1, eps1e-12; moments are task-only.

Pressure uses the unweighted average of per-tensor means over six h tensors
(Ph) or twelve h/z tensors (P2). Thus equal lambda does not preserve the h
contribution independently of the added z contribution; that is the approved
recipe comparison, not a matched-gradient-intensity causal experiment.

Validate all500 MiniPile documents:338 complete2048-token blocks,692224 input
tokens; report excluded1444-token tail. Pool integer counts before dividing.
Collect exact/near-zero activation counts, RMS/L2/moments, all-site logical
products, h/z gate rejection energy, a/m/h/z rows and fully empty aligned8x16
and16x16 tiles, learned LN affine parameters, named weight norms, and every
boundary's gradient interaction/OL1 geometry. No clipping sweep is added.
Retain model checkpoints at0,1,2,4,8,16,32,64,128,256,512,712 and full recovery
states at256,512,712, including final. This inherits the approved Run057
measurement inventory so training-only interaction remains reconstructible.

P2 lowering full-validation loss supports a quality benefit; higher exact-zero
or empty-tile fractions support their own sparsity benefit. Regressions refute
those benefits in the tested regime. Report both axes and paired differences.
One seed and two thresholds are descriptive; neither logical opportunity nor
empty tiles imply measured inference speedup. Tile pressure remains deferred.
The manuscript's site/pressure quality-sparsity question is affected, using the
operational OL1 contract. No manuscript or finding update is requested.

Runtime trainer and diagnostics adapt Run057, with frozen Run004 recipe code
and the previously used HZ topology. Inputs remain unchanged after execution;
infrastructure retries, if needed, receive new attempt records.

The concrete launch and closeout records are in [LAUNCH.md](LAUNCH.md).

## Verified results

P2 raises full-validation loss compared with Ph at both tested thresholds:

| kappa | Ph loss | P2 loss | P2 minus Ph | R_model difference (pp) |
|---|---:|---:|---:|---:|
| 0 | 5.127033 | 5.280096 | +0.153062 | +0.3257 |
| .1 | 5.152346 | 5.330544 | +0.178198 | +0.0262 |

Adding z to the mean-pressure recipe increases z sparsity and empty tiles,
while h loses sparsity. At kappa=.1, empty 8x16 tiles change from 20.704% to
74.828% in z, and from 43.696% to 34.653% in h. The corresponding logical
model opportunity barely changes: 5.1708% to 5.1970%. Thus P2 produces more
structured zeros at z with a quality cost, and does not improve validation
quality in this one-seed comparison. The degradation occurs with no a/m gate
or pressure. The declared pressure-mean normalization remains an interpretation
limit; these results do not isolate the added z gradient from h reweighting.

A and m have zero exact-zero elements and zero empty tiles in all four cases.
Full validation covers all 500 documents, 338 complete blocks and the declared
1,444-token excluded tail. No inference speedup was measured.

The [complete tables](results/tables.md), [source-hashed numerical export](results/results.json)
and [observation with captions](observations/001-p2-versus-ph.md) retain both
tile shapes, integer counts, all-site moments and training gradient summaries.
Both final PDF pages were rendered and visually checked; their source and
figure hashes are in [analysis verification](results/analysis-verification.json).
Tile pressure stays deferred and no manuscript or finding was changed.

## Execution and closeout

Deployed source: 2f2823ca7c889b28a9c6ecd95b1440b09ad5eb34. One Secure Pod,
8l18iz1u8gt2cl, started at 2026-10-03T21:11:53.940Z in AP-IN-1 with four H100
80GB GPUs, 112 vCPU and 1,006GB host RAM, at USD13.96/hour. The local deadline
guard and credential-free remote workload guard were armed. The CLI returned
a non-ISO timestamp during creation; the existing Pod's ISO timestamps were
retrieved from MCP before guard setup. No replacement resource or scientific
input change was needed.

The pinned builder reproduced both canonical token-cache hashes; training
tokenization plus final hash verification took 958.892 seconds. All four
production preflights passed: five timed updates after warmup, median
3.389--3.476 seconds/update, complete validation/diagnostics, checkpoint
roundtrip and no skips. The actual workload used about 59,319MiB GPU memory
per condition. All four scientific workers completed 712 optimizer updates
without overflow or skips; the complete remote verifier passed.

All 268 sealed files passed archive and individual member hashes locally:
4,060,751,094 payload bytes; 3,759,206,034-byte compressed archive,
SHA256 daadd72ad8ab5a33d34220dc507ba972fc46def08fed04b47ea6e4882fa18777.
The local scientific verifier passed all four attempts without rewriting the
sealed cohort record. All 48 model checkpoints and 12 full recovery states are
retained locally. Transfer took 189.703 seconds and local figure QA overlapped it.

Deletion returned successfully at 2026-10-03T22:23:02.011409Z; provider inventory
confirmed absence at 22:23:23Z. The owned deadline guard was disarmed afterward.
Lifecycle duration was about 71.13 minutes: estimated GPU USD16.551, total
USD16.567 using the observed partial billing rate including disk overhead.
Both are estimates below the USD45 limit. Posted billing was only USD2.119
for 545,867 milliseconds, so it is incomplete and is not the final invoice.
See [compute closeout](results/compute-closeout.json).

Concurrent CDR resources, the old EXITED Run052 Pod and the existing shared
network volume were not modified and are outside this run's lifecycle/cost.
