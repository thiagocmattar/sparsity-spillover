# Approved parallel launch

The user explicitly requested fastest completion using available RunPod GPUs and
parallel tests. This supersedes the unapproved one-GPU/$20 sequential proposal.
Scientific configuration is unchanged. Six independent H200 workers cover Base
and the five T2/Ph thresholds concurrently; the RTX 5090 handles comparable final
latency. No distributed optimizer or cross-condition state is introduced.

Created resources and assignments are in `parallel-fleet-001.json` and
`parallel-connections-001.json`. Live prices: H200 $4.59/GPU-hour and Community
RTX 5090 $0.69/hour. Six H200s plus one 5090 total $28.23/hour. Deadline is
2026-09-24 19:01 UTC, approximately six hours after creation; fleet spend limit
$200 including infrastructure retries, storage and retrieval. Every Pod has an
independent workstation stop guard. Stop preserves uncopied artifacts; terminate
only after local hash verification. Existing Run052/shared-volume resources are
untouched. Source/image/runtime/cache identities and retained outputs are as in
the original launch packet. Training-only packages omit unused kernel headers.

Assign H200 Pod a GPUs 0/1 to Base/kappa0; Pod b GPUs 0/1 to kappa0.01/kappa0.05;
Pod c GPU0 to kappa0.1; Pod d GPU0 to kappa0.5. Each runs six complete production
preflight updates with full validation and diagnostics before its condition's
712-update scientific run. `15_parallel_train.py` uses distinct CUDA-visible
devices, records persistent per-worker logs, and checks measured parallel ETC
against the deadline with one hour reserved for retrieval.

Check every minute during setup/preflight, every five minutes during training,
and at the projected completion window if sooner. Report step, loss, throughput
and refreshed ETC. Preserve and halt affected work on nonfinite values, skipped
updates, failed qualification, ten-minute stale progress or deadline overrun.

Focused suite after the parallel launcher change: 11 passed (18.15 seconds).
The previously completed full bootstrap suite had 242 passes. Launch transport
uses current runpodctl 2.14.0/SSH plus MCP lifecycle calls; no API credential is
sent to a Pod. Bootstrap and code setup run concurrently on the five Pods.
