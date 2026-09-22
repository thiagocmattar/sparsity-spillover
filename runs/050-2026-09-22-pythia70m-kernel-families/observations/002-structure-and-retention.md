# Structure, executed work and retained diagnostics

Question: which sparse structure supports the qualified latency result, and
has the agreed diagnostic inventory been retained?

## Method and coverage

Diagnostics follow the immutable final policy; they do not select or retune it.
All338 validation blocks are visited for the dense policy, C, A and C with
skipping disabled at each T2 checkpoint. Base's no-gate route is collected once.
The500-document source,692224 input tokens and1444 excluded tail tokens match
the numerical qualification. All counts below are pooled integers before
division; RMS pools sums of squares. Precision is the actual BF16 execution.

The full inventory includes all-site exact/near-zero counts(.001/.01), RMS/L2,
weight norms, per-row and256-feature histograms, active-feature unions across
1/4/8/16/32 rows, empty-tile counts,2:4 eligibility and lossless overflow counts.
Maximum occupancy and exact excess counts for31/63/127 segment slots are
derived from the complete histograms. Existing training gradient/OL1 records,
checkpoints and cache identities remain retained through the referenced runs;
inference does not reconstruct training gradient interactions.

## Observed structure of the selected C execution

| Kappa | h exact zeros | h mean NNZ/2048 | z exact zeros | z mean NNZ/512 |
| --- | ---: | ---: | ---: | ---: |
|.05|98.322769%|34.349700|93.320496%|34.199061|
|.1|98.918745%|22.144097|96.845088%|16.153151|

Global sparsity hides large layer differences. Mean h NNZ per row, followed by
the selected consumer's padded tensor-core product positions as a percentage
of the layer's ideal dense product count:

| h layer | .05 NNZ | .1 NNZ | .05 padded work | .1 padded work | Route |
| --- | ---: | ---: | ---: | ---: | --- |
|0|96.9653|64.1396|unmeasured native instructions|unmeasured native instructions|dense|
|1|5.3363|3.0159|5.0523%|3.4019%|C,8 rows|
|2|7.7608|3.7748|8.2490%|6.4645%|C,4 rows|
|3|10.0954|8.0538|10.1160%|8.4474%|C,4 rows|
|4|19.3796|14.8734|16.4177%|13.4726%|C,4 rows|
|5|66.5608|39.0070|23.6191%|13.9960%|C,16 rows|

The work percentages include the padded16-row MMA tile; they are not global
zero rates, native instruction counts, removed model FLOPs or measured speedups.
For example, grouping four token rows still computes16 MMA rows. The compact K
list compensates for this padding in the selected layers. z retains the fused
dense route despite its high exact-zero rate.

High2:4 eligibility alone is also insufficient. At h layer1, .05 has2768556
eligible M16/K32 tiles out of2768896; .1 has2768895. Nevertheless, the tested D
decomposition is slower after encoding and overflow processing are included.
The final policy does not discard an overflowing nonzero or impose a row-NNZ cap.

## Work counters and profiles

The actual conversion count/index buffers are checked against independent
operand masks on every diagnostic block. A's scalar products follow the
validated packed counts. C's consumer loop counts include each output tile,
padded K tile and padded row. The retained PTX identifies
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; eight static MMA sites per
warp, four warps and the checked loop counts give the reported dynamic
warp-instruction instances. The product-count identity is asserted by the
reducer. These are source/PTX-derived execution counts, not hardware counters.

The selected C consumers use55--56 registers per thread with zero reported
spills. Compiler metadata, PTX/CUBIN files, CUDA extension binaries, build flags,
cuobjdump resource reports and compilation logs are retained. Separate four-input
graph and eager traces are diagnostic evidence; their instrumented durations
are not used as the reported latency.

**Hardware traffic is unavailable.** Nsight Compute2025.1.1 returned
`ERR_NVGPUCTRPERM` on the first bounded counter probe. The command, CSV error,
application log and result are retained in `artifacts/hardware-001/`. No driver
permission setting was changed. Requested BF16 weight bytes and scanned input
bytes can be derived from the validated metadata, but must not be called
measured DRAM/L2 traffic or memory-transaction savings.

## Retention and resource state

The final reconciliation verifies **972 remote artifact files,269158504 bytes,
across36 attempt directories**, including failed attempts and compiled builds:
zero missing files and zero hash differences. The inventory and its manifest
hash are in [`retention-audit.json`](../results/retention-audit.json).
Raw timings, every block's numerical result, source hashes, all validation
diagnostics, profiles, work counters and failures are local. The original
checkpoints and token-cache identities remain unchanged and locally retained.

Pod `kym4fmsrbsg1s6` remains **RUNNING**, as requested; no Pod or volume was
created or deleted by Run050. The20:16:03UTC provider check reports$0.99/hour.
The approved compute-stop guard remains22:42:28UTC on22September2026
(19:42:28 Sao Paulo); stopping compute retains the Pod's files. Scientific
workers completed at20:10:53UTC, approximately$1.46 of extension GPU time from
18:42:28UTC. Billing continues while the Pod is retained running, within the
approved$3.96 four-hour GPU ceiling plus retained storage.

Source scripts: [`07_diagnostics.py`](../07_diagnostics.py),
[`work_counters.py`](../work_counters.py), [`13_mechanism.py`](../13_mechanism.py),
and [`audit_retrieval.py`](../prelaunch/audit_retrieval.py).
[`mechanism-summary.json`](../results/mechanism-summary.json) links every
underlying structure/work/profile/compiled-PTX source by SHA256.
