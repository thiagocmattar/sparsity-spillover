# Run038: direct h/z strategy port to a/m at 14M

## Execution outcome (appended 19 September 2026)

Preflight failed before the first synthetic output check because the fixture
had 16 rows and the untouched K042 comparison kernel requires a row count
divisible by 32. This is a fixture error, not a numerical or performance result
for the candidate. No full-model smoke or scientific process ran.

All 38 output files were recovered and SHA256 verified locally, including the
traceback and closed worker log. [Run039](../039-2026-09-19-pythia14m-am-port-fixture/README.md)
corrects the fixture to 32 rows, with identical CUDA candidate and full-model
protocol, reusing the same Pod and original deadline/cost cap. The original
prelaunch record below remains for provenance.

Run039 subsequently completed all 15 scientific processes. Its artifacts also
verified locally before the shared Pod was deleted. The provider confirmed no
Pods remain; total shared GPU cost was approximately USD 0.616 plus storage.
See [Run039 closeout](../039-2026-09-19-pythia14m-am-port-fixture/artifacts/closeout/teardown.json).

Status: implemented; design and launch both explicitly approved on
19 September2026. The user also explicitly instructed not to ask again for
RunPod launch permission. Execution remains bounded by90 aggregate GPU-minutes
and USD2 incremental total. No results are claimed before CUDA qualification.

## Question and fixed inputs

[Approved design](../../analyses/024-2026-09-17-h-only-kernel-latency/AM-LOAD-AVOIDANCE-DESIGN.md):
can the final K049 h/z M8/K16 load-avoiding hybrid improve a/m projections?
Reuse Run037's exact Run029 c30 final T7/Pall checkpoint, kappa0.5, seed1234,
step712 and1493172224 training tokens. No training or pressure update occurs.
The original AdamW/data order, gates, lambda=b=1 and checkpoint identities
remain fixed. Weight SHA256:
`f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.

One RTX5090, BF16, batch1,2048tokens, uncached causal inference, full50304
logits, CUDA graphs. Same Python3.12/PyTorch2.11.0/Transformers5.12.1/CUDA12.8
pins. Complete qualification:500documents,338blocks,692224input tokens,
691886prediction tokens and1444excluded tail tokens. Original bounds:
logit atol0.25/rtol0.02/relative-L2 0.02, pooled loss difference0.001.
Runtime seed2801; same64validation timing identities selected by seed2504,
seven paired passes, three fresh processes per mode in deterministic randomized
order. All original numerical failures and timing samples are retained.

## Implementation and comparisons

`candidate/projection.cu` and `site_port.py` adapt K049's eight-real-row hybrid
to separate K128/N384 and K128/N512 linear operations. Support masks avoid
empty K16 steps before requesting weights; eligible <=2-nonzero rows use the
same scalar arithmetic, and other rows use a padded M16 matrix fallback.
No extra gate is applied. Bias/output rounding, h/z fusion, attention,
LayerNorm, residuals and vocabulary head are unchanged.

Five modes: untouched `frozen`; `port-a`; `port-m`; `port-am`; and
`port-am-dense`, which retains the new layout/padding but disables scanning,
zero skipping and scalar replacement. The last mode is not A0. All modes are
measured afresh on the same physical GPU; old absolute timings are not reused.
`01_prepare.py` freezes1383 input/source identities and exact dependency pins.
`02_benchmark.py` reuses Run037's complete validation and timing protocol.
`03_execute.py` runs direct checks, then bounded smokes, then fifteen scientific
processes. `07_reduce.py` rebuilds13440raw candidate/native timing samples and
reports per-process ranges, contrasts and qualification. Effects are not
assumed additive; process spans are descriptive, not confidence intervals.

## Checks and planned GPU scope

- Focused CPU tests:14passed in0.93s. They check all five settings, signed
  contrasts, independent padded/short-row counters, unique output coverage at
  both widths, input identities and the complete randomized process matrix.
- Full bootstrap:242passed in8.29s. Syntax checks pass for19new run Python
  files. An initial recursive bytecode-generation attempt hit Windows long-path
  errors inside the untouched archive; in-memory syntax checks of the new files
  avoid that irrelevant cache-writing limitation. No archived source changed.
- GPU checks have not run at this prelaunch snapshot:32synthetic cases plus
 192captured training-operand cases (first eight retained seed2500blocks,
  six layers,two sites,two modes); each checks outputs and independent counts.
- Smoke:five modes, eight correctness blocks, four timing inputs,two passes.
  Smoke data never become scientific results. Failed shared implementation
  checks stop dependent execution and remain in the recovered evidence.

## Retention and interpretation

Full-validation untimed diagnostics preserve exact/near-zero counts at
0/.001/.01, activation RMS, row nonzero histograms,8x16/16x16 empty-tile
counts, issued/bypassed matrix instructions and scalar products. Port counters
are checked against an independent operand-derived oracle. Source-level weight
element requests are derived from these instructions/products; they are not
measured DRAM traffic. Existing weight norms/checkpoint identities are retained.
Raw timings, per-block output checks, losses, failures, configs, source/runtime
hashes, logs and final checkpoint/cache identity remain recoverable. No new
clipping or training-gradient diagnostics are introduced.

A correct, repeatable full-model improvement over frozen supports this direct
port. Numerical failure, slower execution or a difference unresolved against
process variation does not. A negative result is limited to this checkpoint,
implementation and workload, not a proof that a/m load avoidance cannot work.

## Authorized execution and teardown

Latest prelaunch read: no Pods; existing100GB shared volume unchanged.
Secure RTX5090 stock LOW atUSD0.99/hour, with EU-CZ-1/EU-RO-1 listed. Refresh
before creation. Proposed name `run038-am-load-avoidance-001`, one32GB GPU,
40GB container disk plus15GB Pod volume at `/workspace`, SSH only; no new
network volume. Image remains the verified Run033 digest
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.

Local RTX5070Ti Laptop12GB has adequate nominal memory but differs from the
matched GPU and Linux toolchain. Historical peak allocation3.034GiB fits the
selected32GB device. Provisional cloud ETC35-65min, including setup/compilation
and recovery; warm scientific phase about9-12min. Ninety minutes of GPU at the
read price isUSD1.485, leaving room for temporary storage under the USD2cap.
Use fast container-local venv/compiler cache from the outset; checkpoint,
source bundle, scientific logs and outputs stay on persistent Pod storage.

Set independent Pod/local stop guards at90min, reserve10min for retrieval,
monitor every60s (earlier if ETC is shorter), and report progress, loss/errors,
throughput and refreshed ETC. Investigate nonfinite/error/counter failures,
stale events, disk pressure or projected overrun. Copy all completed and failed
artifacts, verify archive and file SHA256 locally, then delete only the owned
Pod and verify no unintended compute remains. Preserve the existing volume.
