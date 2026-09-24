# Authorized parallel launch

The user confirmed the complete T7/Ph design and explicitly said to run it on
RunPod as fast as possible. Five concurrent training conditions, one GPU each;
a separate RTX5090 for isolated latency measurement. Prefer available US H200s,
with measured H100 fallback if needed. Pinned image/runtime in config.yaml.

Live quotes on 24 September: Secure H200 USD4.59/GPU-hour; Community RTX5090
USD0.69/hour. Five H200s plus one5090 total USD23.64/hour. Operating envelope:
maximum USD160 new-resource spend and six hours from first creation, including
setup/preflights/retries/transfers; record an absolute deadline with allocation.
Allocate40GB container disk plus60GB persistent volume for a single training GPU,
80GB for two or the timing Pod. No new network volume. Approximately11GB final
checkpoint/recovery bytes, plus calibration, complete metrics/logs and receipts.
Hash-verify every artifact locally before deleting its Pod. Deadline guard stops
compute and preserves uncopied artifacts. Existing stopped Pod and shared100GB
volume remain untouched. Account audit found only those existing resources.

Completed local checks: ten focused tests passed (five gate/gradient conditions,
canonical initialization, matched14M/70M recipes, exact data order/coverage,
real OL1 boundaries, checkpoint reload and analytic reach); full bootstrap suite
242 passed. An initial concurrent invocation shared pytest's default temporary
directory and produced two setup errors; the isolated focused rerun passed.
No scientific test failed. CUDA components and complete end-to-end lifecycle
checks run on the assigned cloud hardware before final measurement/production.

Local Run054 T2 calibration was about21s/update on the laptop, with8.24GB
reserved; it is context only, not a new T7 ETC or fit measurement. The user has
chosen cloud speed. Exact T7 production-shaped six-update calibration runs on
every assigned GPU, including complete validation/diagnostics/checkpoint work;
first boundary excluded from forecast. Require finite/no-skipped updates, six
h pressure captures, >=10% VRAM headroom, and forecast1.3xtraining +10min overhead
+one-hour retrieval reserve within the deadline before starting712 updates.
Prior H200 T2 training was aboutone hour/condition; plan2-3hours elapsed including
setup, validation, diagnostics, timing and transfer, pending exact T7 preflight.

Training monitor5min, setup/timing1min; sooner for completion/warning. Warn on
missing process, stale10min events, skipped/nonfinite boundary, wrong capture,
source/cache drift, insufficient disk or ETC exceeding the budget. Long jobs
use nohup/setsid and persistent /workspace logs; workstation stop guards bound
billable runtime. Seal terminal evidence and retain environment freezes, all
raw timing samples, numerical qualification and work-counter evidence.
