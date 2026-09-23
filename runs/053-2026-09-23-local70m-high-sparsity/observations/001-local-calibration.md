# Local full-model calibration

Question: does the local RTX5070Ti Laptop support representative70M kernel
iteration, and does the old adaptation retain its high-kappa response?

Method: unchanged step712 Base and T2/Ph .05/.1/.5 checkpoints, pinned CUDA12.8
and PyTorch2.11 runtime, BF16 B1/T2048/full50304 logits. Five graph implementations
plus a native eager numerical anchor, all resident together. First four retained
training blocks, two randomized paired timing passes, one process per checkpoint.
Profiles use two inputs in a separate instrumented pass. No figure generated.

Result: all80 graph/block comparisons pass, with bitwise logit agreement on
this small prefix. Peak allocation4.923GiB and peak reservation5.502GiB establish
local fit for this comparison. All72 output files are retrieved and SHA256-verified.
After first compilation, individual checkpoint smokes take approximately6seconds.
The complete retry, including cold compilation, took307.1seconds.

The optimized adaptation takes4.706ms at .5 versus native Base5.938ms. At .05/.1
it takes7.430/6.915ms and remains slower than native Base. Its h/z skip-disabled
control takes8.062/8.541/7.257ms at .05/.1/.5. These are pilot timings, not
significance or manuscript claims. The exact table preserves every control.

The separate profiles place the h/z fallback among the largest costs at the
moderate endpoints (about2.87--2.88ms of instrumented GPU kernel duration per
input). This supports investigating its overhead, but does not measure DRAM
traffic or identify one universal hardware bottleneck. Prior Run042 already
tested M8/M16 layouts; repeating that choice alone is not a novel remedy.

Next implementation step: wider active-feature union groups and a separate
tile-support consumer, with complete conversion costs and matched skip-off/
dense/prior controls. Source weights, gates and validation definitions remain
fixed. No full-model new-candidate result exists yet.

Caveats: four training blocks are not all500 validation documents; one process
and eight timing samples per cell do not establish significance. Laptop/WSL
submission and clock variability remain relevant. Per-component batched graph
timings in the following screen will be a different timer, not directly
substitutable for this full-model latency. Local and RTX5090 cohorts stay separate.

Sources: `02_calibrate.py`, `05_reduce.py`, `results/smoke-002-summary.json`,
`results/smoke-002-table.md`, `results/smoke-002-retrieval.json`. The summary
contains source-file hashes. Failed `smoke-001` and the long-path deployment
correction remain retained; neither is erased or treated as a kernel failure.
