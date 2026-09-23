# Proposed Run052 execution

No Pod has been created. The user has approved the adaptive design, while the
new billable envelope in `proposed-launch.json` awaits launch approval.

One Secure RTX5090 at USD0.99/hour, maximum four hours from resource creation,
USD5 incremental budget including running disk charges and transfer reserve.
30GB container +40GB Pod volume; no new/shared network-volume attachment.
Live quote on 23 September 2026 reports LOW availability. Recheck the actual
Pod price before committing spend. Use the pinned image and Run049 environment.
Estimated task ETC is 2--3 hours, with the first decision point after45--75min;
these extrapolate Run050 durations and must be revised after the GPU smoke.

After approval, discover Pods again, create one scoped `run052-*` Pod, record
its exact id/name/create time/price and four-hour UTC deadline in the ignored
lease record. Start `local_stop_guard.ps1` with `Start-Process -WindowStyle Hidden`
and those explicit values. The guard uses the verified current CLI syntax and
refuses a different name. It stops compute at the deadline and retains disk if
retrieval is incomplete. The local machine must remain awake for this guard;
the on-Pod worker separately enforces the workload deadline. These are fail-safes,
not an unconditional provider-level spending guarantee.

Transfer `bundles/input-001.tar.gz`, its receipt `bundle.json`,
`transfer-inventory.json`, `14_verify_archive.py` and `support.py` over native
OpenSSH with the existing registered key. Verify compressed bytes and every
member before extraction into `/workspace/run052`; sibling Run049 and Run052
paths are preserved. Use Run049's sealed `04_setup.sh` to recreate the pinned
runtime and retain its logs/runtime versions. Confirm all input/source hashes.
No full training dataset is transferred: only the literal128-block prefix and
the full validation cache. The five original model checkpoints stay retained
locally and are copied unchanged. No model training occurs.

Use `12_stage.sh --stage STAGE --attempt NAME --deadline-utc UTC` detached with
`nohup`, stdin closed and stdout/stderr saved beneath `/workspace`. Stage order:

1. `smoke` with attempt `smoke-001`: five actual reference checkpoint/model
   implementations, four training blocks, target CUDA graph/numerical checks.
2. `references` / `references-001`: 15 fresh processes, full338-block validation,
   interleaved14M/70M conditions; reduce with `10_reduce.py --attempt
   references-001 --sparse-mode prior_c_graph`.
3. `screen` / `screen-001`: GPU operator/boundary/dynamic-graph checks, then
   complete component screen. Keep all failures, binary/PTX/resource reports,
   raw timing and source hashes. Stop before screen if no E/F path qualifies.
4. `05_select.py --screen screen-001-screen`: training-only component choice.
   Inspect evidence and change course if justified, within the initial8 plus
   at most8 follow-up configurations. Append new source versions/attempts;
   do not overwrite executed candidates or use validation to choose dispatch.
5. `training` / `training-001`: four-block candidate smoke, then128-block full
   model qualification at both endpoints. `07_freeze.py --selection
   selection-training.json --attempts training-001-c24-r1 training-001-c25-r1`
   seals only if both endpoints improve over prior C and every strong dense
   reference, all required controls qualify, and source identities agree.
6. `final` / `final-001`, then `diagnostics` / `diagnostics-001` and
   `10_reduce.py --attempt final-001`. Stop with a documented negative result
   if no candidate warrants final promotion. Never manufacture a larger drop
   by slowing kappa=.05.

Monitor every60seconds, reporting progress, latest finite loss, stage throughput
and refreshed ETC. Compile/setup stages have no loss: report not applicable.
Warn on numerical failure, less than8GiB GPU headroom, less than5GB disk,
10minutes without progress or an ETC crossing the remaining budget. Workers
stop20minutes before the Pod deadline so retrieval has its reserved window.

Collect using `13_collect.py --tag TAG` after workers stop writing. Transfer the
output archive/inventory/receipt and verify with `14_verify_archive.py`; keep
the verification report locally. Retain all checkpoint/cache identity, full
70M activation and norm statistics, row/group occupancy, work counters, raw
timing, qualification failures, profiles, compiled evidence and source versions.
The14M kernel is frozen; its historical diagnostics remain linked and retained,
while new same-session full numerical and timing measurements are collected.

Notify the user at decision points and completion, and keep the GPU available
for guidance within the approved period. No silent extension. Stop at the hard
deadline; terminate the Pod after verified artifact retrieval and closeout or
user guidance. Confirm no unintended billable resources remain, preserving the
pre-existing shared volume. An incomplete transfer must not trigger deletion.
