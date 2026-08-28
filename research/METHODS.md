# Scientific Methods

## Manuscript relationship

The living draft under `manuscript/` supplies the current paper framing and
formal derivations. This file is the compact operational contract. A substantive
difference between them is an explicit design choice, not an editorial error to
hide. Record which definition an experiment executes.

## Model construction

For Pythia pretraining, load the pinned Hugging Face architecture config and
construct parameters with `AutoModelForCausalLM.from_config`. Keep parameters
and AdamW state in FP32; BF16 may be used for CUDA autocast. Loading released
weights changes the task to continuation or fine-tuning.

Pythia/GPT-NeoX has parallel attention and MLP residual branches. Sites `a` and
`m` are the two branch LayerNorm outputs, not one shared residual tensor.

## Canonical activation ports

| Site | Exact port | Shape | Downstream product |
| --- | --- | --- | --- |
| `a` | Attention-branch LayerNorm/gate output | `[B,T,D]` | fused QKV projection |
| `m` | MLP-branch LayerNorm/gate output | `[B,T,D]` | MLP W1/up projection |
| `h` | MLP nonlinearity/gate output | `[B,T,M]` | MLP W2/down projection |
| `q_pre` | Query immediately before partial RoPE | `[B,H,T,d]` | QK after RoPE |
| `k_pre` | Key immediately before partial RoPE | `[B,H,T,d]` | QK after RoPE |
| `q_post` | Query immediately after partial RoPE | `[B,H,T,d]` | actual QK operand |
| `k_post` | Key immediately after partial RoPE | `[B,H,T,d]` | actual QK operand |
| `v` | Value from fused QKV projection | `[B,H,T,d]` | PV product |

PRE-RoPE zeros cannot be counted as actual QK operand zeros because RoPE may
rotate sparse coordinate pairs into dense ones.

## Topology registry

| ID | Active gate sites |
| --- | --- |
| `A0` | none; stock GELU at `h` |
| `A1-H` | `h` |
| `A2` | `m`, `h` |
| `A3` | `a`, `m`, `h` |
| `A4-Q` | `a`, `m`, `h`, `q_post` |
| `A4-K` | `a`, `m`, `h`, `k_post` |
| `A4-V` | `a`, `m`, `h`, `v` |
| `A5-QK-PRE` | `a`, `m`, `h`, `q_pre`, `k_pre` |
| `A5-QK-POST` | `a`, `m`, `h`, `q_post`, `k_post` |
| `A6-PRE` | `a`, `m`, `h`, `q_pre`, `k_pre`, `v` |
| `A6-POST` | `a`, `m`, `h`, `q_post`, `k_post`, `v` |

A topology chooses ports only. The gate operator, threshold, optimizer, pressure
method, pressure sites, and pressure weight remain separate.

## Gate operators

ReLU is standard elementwise `max(x, 0)`. For an active `h` site it replaces
GELU; it is not applied after GELU.

One-sided threshold:

```text
G+_kappa(x) = x when x >= kappa, else 0
```

Symmetric threshold:

```text
Gpm_kappa(x) = x when abs(x) >= kappa, else 0
```

`kappa` is finite and nonnegative. Equality survives. The comparison is
detached, so surviving inputs receive identity input gradient and rejected
inputs receive zero input gradient. Exact zeros do not make a dense kernel skip
work automatically.

The bootstrap preserves the source implementation's one-operator-per-topology
contract. Mixed one-sided MLP and symmetric attention gates require a new
approved implementation and tests.

## Activation pressure

For captured tensors `A_j`, first average absolute values within each tensor,
then take an unweighted mean across tensors:

```text
L1_activation = (1/J) * sum_j mean(abs(A_j))
```

This gives every site/layer tensor equal weight even when tensor widths differ.
Changing to element-count weighting is a scientific change.

### Naive L1

```text
objective = L_task + lambda * L1_activation
```

The combined gradient is globally clipped to norm 1.0 and consumed by AdamW.

### OL1

At each accumulated optimizer boundary:

1. Accumulate task gradients and pressure gradients separately.
2. Globally clip the task gradient to norm 1.0.
3. Let AdamW consume only that task gradient and update its task-only moments.
4. Reconstruct the bias-corrected task direction and precondition the pressure
   gradient using AdamW's task second moment:

```text
d_task = m_hat / (sqrt(v_hat) + adam_eps)
d_pressure = g_pressure / (sqrt(v_hat) + adam_eps)
```

5. Compute one global dot product. Only when it is negative, remove the
   conflicting component:

```text
d_safe = d_pressure - <d_task,d_pressure> / (||d_task||^2 + eps) * d_task
```

6. Compute and cap the weighted pressure/task direction ratio:

```text
raw_ratio = lambda * ||d_safe|| / (||d_task|| + eps)
scale = min(1, step_budget / (raw_ratio + eps))
```

7. After AdamW, apply for each parameter group:

```text
theta <- theta - learning_rate * lambda * scale * d_safe
```

The geometry excludes decoupled weight decay and is computed before per-group
learning rates. `step_budget` is a relative trust budget, not a target sparsity.

## Reference optimization recipe

- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`.
- Global gradient clipping at L2 norm `1.0` immediately before AdamW.
- BF16 CUDA autocast; FP32 parameters and optimizer state.
- Zero hidden and attention dropout.
- Linear warmup for `ceil(0.01 * max_steps)`, followed by cosine decay to 10%
  of the peak learning rate.
- Global batch is stated in sequences and input tokens. The historical baseline
  used 128 × 2,048 = 262,144 input tokens per update; microbatch and
  accumulation are hardware decisions that must preserve the approved global
  batch.

## Post-hoc clipping

At evaluation only, replace selected activations with exact zero using one of:

- absolute cutoff `abs(x) <= t`;
- a requested absolute-value quantile;
- `multiplier * RMS(A)` computed per captured tensor and forward pass.

Every frontier includes its own zero-threshold row. Report paired loss as
`loss(t) - loss(0)` from that same sweep. Joint and site-specific clipping answer
different questions. Clipping does not identify a trained threshold topology or
causally attribute an effect to one site.

## Interpretation limits

- Differentiable pressure may move mass near zero without producing exact zeros.
- Near-zero mass depends on its threshold.
- Logical zero products are not removed FLOPs or measured acceleration.
- Training-time gradient interaction cannot be reconstructed post hoc from a
  checkpoint.
- A last training minibatch cannot substitute for a full named diagnostic.
