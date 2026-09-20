# Pilot closeout, 2026-09-05

**Decision: do not start the full optimization search yet.** P0 passed
primitive checks on both GPUs and complete A0-14M validation, but failed the
fixed A1-H-14M logit gate. Preserve that failed baseline and qualify a
documented correctness repair before sealing P0 or scoring K001.
No numerical tolerance, checkpoint, gate, evaluator or root kernel was changed.

## Spend and resources

| Pod/stage | Rate | Lease elapsed, including overhead | Estimated GPU cost | Final state |
| --- | ---: | ---: | ---: | --- |
| RTX 5090, `ivt0noruc0l142` | $0.69/h | ~22m41s | $0.261 | Deleted after 57-file verification |
| H100 Community allocation | $2.69/h quoted | No allocation | $0 | No matching stock |
| H100 Secure, `ziisaqda7des80` | $3.49/h | ~18m50s | $1.096 | Deleted after 32-file verification |
| Total | — | ~41m31s billed lease envelope | **$1.357** | **Zero Pods/endpoints** |

Lease estimates use the local pre-create timestamp through confirmed deletion,
so include a small conservative control-plane margin. Approximate prorated
Pod disks and existing storage bring the all-in estimate to **$1.38**.
The account balance changed from $42.6340163325 before launch to
$41.3113263139 at closeout: **$1.3226900186 reflected so far**. The hourly
resource billing API returned no posted records yet; do not call that zero
cost or claim the estimate is a settled invoice. No new paid agent API used.

Reserve **$1.40** against the study until billing settles: at least **$38.60
of the $40 study envelope remains**, with no further GPU commitments.
Only pre-existing volume `9luykg5yc3` (100 GB, EUR-IS-1) remains. The account
reports $0.01/hour current spend for storage, not GPU compute. Both local
guard processes were stopped after confirmed Pod deletion. Deadline firing
was not tested; the guards were armed/alive, while teardown was explicit.

## Measured ETC and what remains unknown

| Work | Measured duration / outcome |
| --- | --- |
| RTX upload, 5.83 GB | Approximately 15 minutes, overlapped setup/primitive checks |
| RTX P0 compilation | 28.745 seconds |
| RTX A0-14M paired full validation | ~5.56 seconds, all 338 blocks |
| RTX model loop | 14.67 seconds to A1-H numerical stop; **not eight-model ETC** |
| H100 default SFTP upload | Reset after several minutes; no successful payload |
| H100 compressed legacy-SCP retry | Nine seconds for the 3.76 MB bundle and scripts |
| H100 P0 compilation | 40.546 seconds |
| H100 setup + primitives + upstream control | 6m33s, excluding earlier transfer issues |
| 70M/410M calibration, all-36 final suite, search candidates | **Not measured: blocked by correctness gate** |

Most pilot lease time was setup, transfers and monitoring/retrieval, not
benchmark execution. The RTX checks finished much faster than their conservative
90-minute worker budget. The original lease limits were safety ceilings,
not predicted experiment durations. H100 evidence was complete at 15:46:05;
monitoring/inspection/retrieval and teardown added about 5m33s (~$0.32).
Future short tests should emit a completion notification or use calibrated
completion-window checks; a ten-minute poll is excessive for a seconds-long loop.

The prior full-study **32 RTX-5090 hours + 4 H100 hours** remains a bounded
allocation, not measured ETC: **$32.84 GPU-only** at Community quotes, or
**$36.04** if all H100 hours require Secure at $3.49/h. Calibration is included,
not an extra allowance. Storage, retries, transfers and any incremental agent
charges must still fit the $40 ceiling. Community H100 stock failed in this
pilot; do not assume the cheaper rate is deployable. Do not spend the reserve
on additional search hours without protecting final evaluation first.

## Next steps and test playbook

1. Diagnose P0 accumulation/rounding on development inputs with intermediate
   linear outputs and a high-precision reference. The current two-block
   diagnostic is evidence, not a complete root-cause proof. Preserve P0's
   failed source identity; record any repair as a new correctness revision.
2. Keep existing numerical gates fixed. Run unit tests, real retained-model
   smoke, 48 primitive/sanitizer checks, then the unchanged eight-checkpoint
   complete-validation gate. No optimization-score search until this passes.
3. Use measured 70M/410M timings to protect the all-36 final evaluation,
   component ablations and matched frozen-kernel H100 comparison; only then
   select final candidate limits and present defensible completion estimates.
4. Seal P0, strongest compiled/graph dense baseline, exact attention contract,
   evaluator and trajectory provenance before K001. Attention QK/PV is still
   dense in the current adapter. The approved argument chain remains a
   hypothesis, not a validated paper result.

For infrastructure reuse: prepend both CUDA and venv bin directories to PATH;
check actual allocation rather than aggregate catalog availability; await
transfer exit zero and enforce archive/file hashes before extraction/execution;
use `scp -O -C` on the demonstrated H100 transport; maintain an independent
local billing guard plus bounded detached workers; retrieve and verify before
deleting. Failed attempts, sources and decisions remain in `launch-control/`.

Full bootstrap: **242 passed**, including a rerun before H100 launch. The
real CUDA and model gates above, not those CPU tests, determine scientific
qualification. No manuscript text, trained models, source-run records or
untuned-checkpoint partition were changed. See the
[observation and raw evidence links](observations/001-calibration-gates.md).
