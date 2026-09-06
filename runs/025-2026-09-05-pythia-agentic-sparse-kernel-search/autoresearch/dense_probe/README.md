# Matched dense execution probe

Development diagnostic only. This adds no kernel, changes no trained topology,
and does not modify the original P0/evaluator. No local GPU or cloud execution
is performed by its CPU tests. Runtime measurements require a separate GPU
invocation by the run owner, inside the approved Run 025 budget.

`probe.py` compares native eager dispatch with CUDA Graph replay and optional
`torch.compile` full-graph `reduce-overhead`/`max-autotune`. Each invocation
stages the **actual rotating input** into one common persistent input buffer
outside the timer for every mode. The synchronized host timer includes the
entire model forward and all vocabulary logits, causal attention, exact trained
gates, and LM head. No activations or logits are cached across inputs. CUDA
event timing is auxiliary. Capture/compile/load/staging time is separately
reported. Run the sparse route with the same capture and staging treatment
before using this as a matched baseline; graph-vs-eager differences are not
sparse acceleration.

By default, quality/timing uses the first 16 pinned development training blocks,
three paired passes, and all fixed Run 025 logit/loss bounds. The fastest mode
is only provisional. `--full-validation` additionally evaluates all 338 complete
validation blocks (500 documents; 1,444-token tail) for every development-valid
mode and pools summed cross entropy across shifted predictions. It never uses
validation timing to select a mode. Interior-kappa checkpoints are rejected.
Unsupported compile/capture modes are retained as failed setup entries, not
silently replaced by eager execution. Do not relax numerical gates to rescue
an otherwise fast compiler configuration.

Example on the already provisioned GPU, from the repository root:

```bash
timeout --signal=TERM --kill-after=30s 1200s "$VENV/bin/python" -u \
  runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/dense_probe/probe.py \
  --condition 14m/a1h --attempt graph-compile-001 --seconds 1100
```

`--modes native graph` is the cheaper first screen (no Triton compilation).
`--sdpa-backend flash` forces FlashAttention instead of automatic SDPA dispatch
in a **separate attempt**; unsupported hardware/shapes are failures. Compare
both only on development inputs and freeze the selection before final tests.
Within a forced-Flash attempt the reference is also Flash: that attempt alone
does not establish numerical agreement with the automatic-backend reference.
If those backends differ, the final evaluator must also gate their paired
logits/loss before promoting the backend choice.
For 410M, load just one checkpoint per process. `--seconds` is cooperative:
the outer process-group timeout is mandatory because compilation can block.
No billing guard or resource teardown is implemented here.

## Attention audit

The current canonical `_attention_forward` **does not force eager scores or
probability materialization**. In `src/sparsity_research/pythia.py`, post-RoPE
Q/K and V are gated before the selected Transformers attention interface. That
interface uses `config._attn_implementation`; calibration explicitly requests
`sdpa`. Transformers 5.12.1's SDPA interface calls PyTorch's fused dispatcher
with the same gated operands, scaling, causal flag, and zero inference dropout.
The context is then concatenated, gated at z, and projected by W_o. Thus the
exact fused dense attention path already exists and must be the comparator.

The default implementation name does not prove which GPU kernel actually ran.
`native-attention-profile.json` records attention/flash/softmax/bmm operators
from one untimed native forward to verify dispatch. Capture instrumentation
that requests actual attention probabilities and logical products is a separate
diagnostic and must not be left on the timed model. Flash SDPA changes numerical
reduction order relative to materialized eager attention; the unchanged model
quality gates still apply.

Dense fusion is not sparse attention. A subsequent exact sparse attention
component must consume the gated post-RoPE operands and preserve causal softmax
normalization: zero Q gives a valid-prefix average, zero K still contributes
normalization mass, and zero V does not remove its key. Native SDPA already
handles all these cases correctly. Existing gates/ports need not be rewritten
to obtain fused attention; any future fused gate/attention specialization is
new optimization code with its own correctness and component ablation evidence.

Primary API references checked for the pinned Torch 2.11 line:
[CUDA Graph semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)
and [torch.compile modes](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html).

CPU verification:

```powershell
.venv/Scripts/python.exe -m pytest -q runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/autoresearch/dense_probe/test_probe.py
```

Tests cover equal staging/real input rotation, full-output shape inside the
timer, pooled shifted-token loss, failure of a loss-preserving wrong-logit
offset, actual A7 post-RoPE gated operands reaching SDPA, causal no-mask dispatch,
and zero-query causal prefix averages. They cannot verify CUDA capture,
compiler correctness, or speed; those remain explicit GPU gates.

Local result: **5 tests passed** in 2.92 seconds; `compileall`, CLI `--help`,
and scoped whitespace checks also passed. No GPU result exists at this handoff.
