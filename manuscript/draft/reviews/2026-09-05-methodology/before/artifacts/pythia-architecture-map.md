# Pythia architecture map

## Question

What is the intervention-neutral forward graph of one shared
Pythia/GPT-NeoX transformer block?

## Method and coverage

The diagram transcribes the parallel-residual block in `research/METHODS.md`
and `manuscript/methodology.tex`. It covers the attention, FFN, and identity
residual paths from `H_l` to `H_{l+1}`. The attention path includes its branch
LayerNorm and activation site `a`, fused QKV projection, RoPE, attention operand
sites `q`, `k`, and `v`, scaled dot-product scores and softmax in one operator,
probability site `p`, probability-value multiplication, attention-result site
`z`, and output projection. The FFN path includes its separate branch LayerNorm
and sites `m` and `h` around `W_1` and `W_2`.

Input embeddings and the final language-model head are intentionally outside
the figure's scope. Pre-RoPE query and key tensors are also omitted; `q` and `k`
denote the post-RoPE operands consumed by `QK^T`, without pre/post suffixes in
the graphic.

The delivered standalone PDF is intentionally rotated 90 degrees clockwise for
the requested manuscript orientation.

## Visual encoding and caption

Every visible graph node is a rounded rectangle. A compact right-aligned header
key uses colored dots plus text for the three intervention-relevant semantics:
green activation sites, purple learned matrix weights, and gray operators.
Residual states remain dark blue and are identified directly by `H_l` and
`H_{l+1}`. Arrows denote tensor flow.

**Figure caption.** One shared Pythia/GPT-NeoX block. The residual stream
`H_l` branches in parallel through attention, the FFN, and the identity path.
In attention, `LN_a -> a -> W_QKV` produces query, key, and value paths; RoPE
precedes the `q` and `k` operands used by a compact `QK^T`/softmax operator,
which produces activation site `p`, while `v` enters the probability-value
product. Its result is exposed as activation site `z` before `W_o`. In the FFN,
`LN_m -> m -> W_1 -> h -> W_2`. The three paths merge at `H_{l+1}`.

## Result

This is a structural schematic, not an empirical result. It remains
intervention-neutral so later variants can overlay gates, pressure targets, or
diagnostic sites without redrawing the base graph.

## Caveats

- `LN_a` and `LN_m` are separate parallel-branch LayerNorms.
- `q` and `k` are the post-RoPE QK operands even though the compact figure does
  not use `pre` or `post` suffixes.
- Biases, QKV reshape/split mechanics, head concatenation, dropout modules,
  tensor shapes, and the training loss are omitted for legibility.
- The `h` block denotes the FFN activation port; a later intervention overlay
  may specify the executed activation operator.

## Source

- Editable source: `pythia-architecture-map.tex`
- Generated artifact: `pythia-architecture-map.pdf`
- Scientific contracts: `../../research/METHODS.md` and `../methodology.tex`
