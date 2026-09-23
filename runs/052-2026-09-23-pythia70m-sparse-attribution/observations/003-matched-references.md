# Matched reference matrix before new kernel selection

Question: does the retained Run051/Run050 C policy already establish the
requested larger positive kappa-neighbour latency drop at70M?

Method and coverage: one RTX5090, the pinned BF16 runtime and unchanged retained
checkpoints; three fresh process rounds per condition, reversed condition order
in the second round. Each process numerically checks all338 complete MiniPile
validation blocks (500 documents;691,886 predicted tokens;1,444-token excluded
tail). Timings cover the same64 inputs with seven passes per process. All modes
qualified in all15 processes. Latency below is the geometric mean in ms.

| Model / condition | Native PyTorch graph | Strong dense policy | Retained sparse policy |
|---|---:|---:|---:|
|70M Base|1.681633|1.286617|1.285794 (dense fallback)|
|70M T2/Ph .05|1.749723|1.283211|1.217876|
|70M T2/Ph .1|1.746349|1.282619|1.203205|
|14M T2/Ph .05|.713239|.503575|.586727|
|14M T2/Ph .1|.714087|.505719|.571974|

Result: Delta14=14.753us and Delta70=14.671us. Their difference is -.082us,
with paired crossed-process/input marginal95% interval [-4.401,4.680]us.
The requested strictly larger70M drop is not established. Both70M endpoints
beat native Base, by1.381x and1.398x, and beat their same-checkpoint strong dense
controls by65.335us and79.414us. These sparse routes remain h-only; z is dense.
The dense-adjusted cross-size response is -2.819us [-6.029,.588]us.

Native full-validation BF16 loss is4.1076898430 (Base),4.2000801410 (70M .05),
4.1900605993 (70M .1),5.1355107923 (14M .05),5.1523267297 (14M .1). All process
replicates agree. These are numerical-qualification losses; historical canonical
FP16 quality records are separate and unchanged.

Caveats: this is reference evidence, not a result for new E/F candidates. Strong
dense14M is faster than the frozen historical K050 kernel; retain that control
alongside the requested historical-kernel Delta14 comparison. The intervals
describe this process/input sample on one device, not training seeds or a GPU
population, and do not have simultaneous family-wise coverage. Dense-adjusted
effects do not replace the requested raw Delta criterion. No manuscript claim
is changed. No figure was generated.

Source scripts: `06_benchmark.py`, `09_execute.py`, `10_reduce.py`, `effects.py`.
Evidence: `results/references-001-summary.json` and the15 attempt folders under
`artifacts/references-001-*`. Retrieval of190 archive members passed all sizes
and SHA256 checks; see `prelaunch/retrieval-references-001.json`.
