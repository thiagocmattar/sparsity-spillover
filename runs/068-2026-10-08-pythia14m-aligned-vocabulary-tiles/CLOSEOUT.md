# Run068 execution closeout

Both approved fresh trainings and the separate RTX 5090 benchmark completed.
Head κ is 0.1/0.5; h/z κ remains 0. The gate, pressure, optimizer, initialization,
data order and numerical acceptance criteria stayed at their approved values.
The [scientific report](results/REPORT.md) contains the loss, occupancy, latency,
qualification tables and both publication PDFs.

## Verification and retained evidence

Each model completed all 712 updates with no overflow or skipped optimizer
boundaries. All scheduled native validation and diagnostics cover all 500
MiniPile validation documents: 338 complete 2,048-token blocks, 692,224 input
tokens and 691,886 predictions; the 1,444-token tail is excluded.

The training archive contains 150 retained files and 2,144,698,798 payload bytes.
Its 1,992,908,506-byte compressed archive has SHA256
`f49d0b688660670857f4b6deeb3cdf0e57b2e54704c388bb865c3326abf7e6e3`.
All archive/member hashes and terminal scientific counts were verified locally
before deleting the training Pod. The 24 scheduled model checkpoints, two
calibration checkpoints, six recovery states and complete training/OL1
interaction diagnostics remain local. Weights and recovery states are excluded
from Git; byte inventories and scientific measurements are versioned.
Sources: [training verification](artifacts/verification.json),
[inventory](transfer/training-inventory-001.json),
[retrieval](prelaunch/retrieval-training-001.json).

The benchmark used the actual step-712 endpoints. All 81 sealed input files
match their original hashes. The 63 cloud component checks, both eight-block
smokes and six full fresh processes passed. Full qualification includes 6,084
native-reference logit comparisons; all have maximum absolute error zero and
pooled loss delta zero. All 2,304 standalone head-graph comparisons are exact.
All six processes use GPU UUID `b596bab9-9965-95f9-434d-4d5496057625`,
Torch 2.11.0+cu128, Transformers 5.12.1, NumPy 2.5.0 and CUDA 12.8.
The original numerical bounds remain unchanged.

Each timing boundary has 64 fixed inputs × seven randomized paired passes ×
three modes × three processes per model. Synchronized host wall time is primary;
separate CUDA-event replays agree on the direction of every aggregate comparison.
The real-operand instrumented consumer is checked over all 338 blocks in
replicate one of each model, for both custom modes. The empty gate/consumer tile
fractions and skipped/potential MMA fractions match exactly. Training and the
native inference baseline execute dense matmuls; only the custom sparse head
bypasses MMA. All h/z projections remain native dense in this benchmark.

The benchmark archive contains 83 result files and 5,418,621 payload bytes.
Its 552,059-byte compressed archive has SHA256
`7b6db74d054d70192e2ca254afad6bbfb1e31a60f9812b1c2a5105e9032d0d0f`.
All members were hash-verified, followed by local cohort verification, before
deleting the benchmark Pod. Sources: [cohort verification](inference/artifacts/verification-001.json),
[inventory](inference/transfer/inventory-001.json),
[retrieval receipt](inference/transfer/receipt-001.json),
[component checks](inference/prelaunch/component-cloud001.json).

## Infrastructure and teardown

| Stage | Owned Pod | Provider start (UTC) | Deletion acknowledged (UTC) | Quoted whole-Pod rate |
|---|---|---|---|---:|
| Two concurrent fresh trainings | `dxnmrzchgtaxbr` | Oct 8, 22:33:38.379 | Oct 8, 23:43:03 | $7.98/h |
| One RTX 5090 benchmark | `m3mw1up56hjdh7` | Oct 8, 23:45:32.322 | Oct 9, 00:19:48 | $1.19/h |

Both deletions returned HTTP 204 and were confirmed by refreshed provider
inventories. Both owned workstation stop guards were disarmed after retrieval
and confirmed deletion. The stages did not overlap. Final inventory contains
only the pre-existing EXITED Run052 Pod `qa00sm05lmrs7w` and the pre-existing
100 GB network volume `9luykg5yc3`; neither was changed. Sources:
[training teardown](prelaunch/teardown-training-001.json),
[benchmark teardown](prelaunch/teardown-benchmark-001.json),
[final resource inventory](prelaunch/final-resource-inventory.json).

The aggregate quoted-rate estimate is **$9.93**: $9.9111 compute plus $0.0193
running storage, using actual provider-start-to-deletion intervals and a
730-hour month. The approved maximum was $60. This estimate is not an invoice.
The posted billing snapshot totals $6.3160 for training and has no benchmark
records yet; it differs from the elapsed-time estimate and is not treated as
final billing. No cause for that difference is assumed. Foreign retained
resources are excluded. [Cost calculation and posted records](prelaunch/cost-closeout.json).

Training deployed source commit `f69ac644166de774c6a1ecc9abf1646105aa69a9`.
The sealed benchmark deployed commit `35a4c2b7d470191f52e439df961ab77c9b5daec2`.
Later documentation/retrieval commits do not change the 81 sealed input bytes.
Sources: [training bundle](prelaunch/source-001.receipt.json),
[benchmark upload](inference/prelaunch/upload-benchmark-001.json),
[trained input inventory](inference/provenance/trained-upload-inventory.json).

Initial Norway SFTP uploads were too slow. Operational attempts
[002](inference/prelaunch/attempts/002/README.md),
[003](inference/prelaunch/attempts/003/README.md) and
[004](inference/prelaunch/attempts/004/README.md) record transport-only retries.
The encrypted runpodctl transfer resumed the same sealed bundle, then verified
its hash and every input file before detached setup/qualification. No scientific
condition was executed from partial bytes. Private transfer codes and logs are
excluded from Git and the scientific archive. The successful detached pipeline
log is retained locally under `inference/prelaunch/`; its exit status is zero.

## Reproduce the reductions

The trained evidence is immutable. Reducers and figures require the verified
transfer receipts. From the repository root, using the pinned local environment:

```powershell
$run068 = 'runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles'
.venv/Scripts/python.exe -X utf8 "$run068/inference/05_verify.py" --tag 001
.venv/Scripts/python.exe -X utf8 "$run068/inference/20_analyze.py" --tag 001
.venv/Scripts/python.exe -X utf8 "$run068/19_figures.py"
.venv/Scripts/python.exe -X utf8 "$run068/21_summary.py"
```

For rechecking the compressed benchmark archive, `08_retrieve.py` requires an
absolute `--archive` path because its byte-record root must also be absolute.
The first local relative-path invocation failed before extraction; the corrected
absolute-path invocation verified all 83 files without changing any evidence.
Publication outputs are PDF only; both PDFs were rendered and visually reviewed.
No manuscript edit or finding promotion is included.
