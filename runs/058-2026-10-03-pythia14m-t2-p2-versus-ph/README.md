# Run 058: 14M T2/P2 versus T2/Ph

Status: implementing the explicitly authorized four-condition comparison.

On 3 October 2026 the user requested T2 (one-sided gates at h,z only), P2
(pressure at h,z), compared with Ph, and explicitly authorized RunPod execution
following Run057 principles: “Beleza, pode rodar na run. Pode seguir nos mesmos
princípios da run anterior. Vai lá.” This authorizes design, implementation,
launch and verification without another confirmation.

Question: does expanding OL1 pressure from h to h,z improve the quality and
structured activation sparsity of the same T2 gates? Fresh T2/Ph and T2/P2 are
trained at kappa=0 and .1 in four concurrent independent workers. The only
within-threshold scientific change is the pressure target set. Neither a nor m
has a gate or pressure. h replaces GELU, z is concatenated PV before W_o;
equality survives the detached one-sided comparison in all six layers.

Match Run057 random step-zero bytes, pinned architecture/tokenizer and MiniPile
revisions, seed1234, exact realized schedule, 712 updates, global batch1024,
MB32/GAS32, T2048, dynamic FP16, FP32 parameters/moments, AdamW .9/.95,
eps1e-8, weight decay.1 excluding bias/norm, task clipping1, peak LR.001,
minimum.0001, warmup.01 and the pinned GPT-NeoX pre-step schedule mapping.
OL1 weight1, relative trust budget1, eps1e-12; moments are task-only.

Pressure uses the unweighted average of per-tensor means over six h tensors
(Ph) or twelve h/z tensors (P2). Thus equal lambda does not preserve the h
contribution independently of the added z contribution; that is the approved
recipe comparison, not a matched-gradient-intensity causal experiment.

Validate all500 MiniPile documents:338 complete2048-token blocks,692224 input
tokens; report excluded1444-token tail. Pool integer counts before dividing.
Collect exact/near-zero activation counts, RMS/L2/moments, all-site logical
products, h/z gate rejection energy, a/m/h/z rows and fully empty aligned8x16
and16x16 tiles, learned LN affine parameters, named weight norms, and every
boundary's gradient interaction/OL1 geometry. No clipping sweep is added.
Retain model checkpoints at0,1,2,4,8,16,32,64,128,256,512,712 and full recovery
states at256,512,712, including final. This inherits the approved Run057
measurement inventory so training-only interaction remains reconstructible.

P2 lowering full-validation loss supports a quality benefit; higher exact-zero
or empty-tile fractions support their own sparsity benefit. Regressions refute
those benefits in the tested regime. Report both axes and paired differences.
One seed and two thresholds are descriptive; neither logical opportunity nor
empty tiles imply measured inference speedup. Tile pressure remains deferred.
The manuscript's site/pressure quality-sparsity question is affected, using the
operational OL1 contract. No manuscript or finding update is requested.

Runtime trainer and diagnostics adapt Run057, with frozen Run004 recipe code
and the previously used HZ topology. Inputs remain unchanged after execution;
infrastructure retries, if needed, receive new attempt records.

The concrete launch and closeout records are in [LAUNCH.md](LAUNCH.md).
