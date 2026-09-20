# opt001: native a/m and attention

Combine the two replacements indicated by the completed initial diagnostic
cells and independent traces. T7/Ph native-attention r1 saved approximately
0.11ms and native-a/m r3 approximately0.09ms against their paired untouched
kernel; these preliminary conditional effects are not assumed additive.
The diagnostic matrix continues unchanged. No optimization timing has been
observed before defining this candidate.

Keep the original fused normalization/gates, RoPE/gates, and sparse h/z
implementation. The same policy executes both checkpoints. No weights,
thresholds, precision, inputs, or numerical bounds change. Select using only
the fixed16 training-development blocks after diagnostic completion.
