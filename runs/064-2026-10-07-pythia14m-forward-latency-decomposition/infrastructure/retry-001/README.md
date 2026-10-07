# Diagnostic shape retry

Attempt full-k0p1-r1-001 qualified all eleven original/mapped/eager paths on
338 blocks with exactly matching logits/loss, and produced timings and traces.
The leaf then failed in internal.py's untimed density reconstruction: h/z had
been flattened for CUDA operands, while CornerTileThreshold requires B/T/D.
No internal-cycle artifact completed. The failed result and every prior log,
trace and numerical/timing artifact remain immutable and are retrieved first.

profile.py wraps ONLY kernel.gated_value for a 2D corner-gate diagnostic input:
add the singleton batch axis, execute the same helper, remove the same axis.
The timed production kernels, mapped forward, thresholds, checkpoints, CUDA
sources, numerical criteria, coverage and original executable are unchanged.
controller.py repeats all six fresh full processes under tag002 on the same
physical GPU and original lease/deadline. No installation or source-code/kernel
recompilation change is needed. No model/diagnostic scientific input is corrected;
this repairs a dimensional adapter before any internal measurement is accepted.

Postprocessing separately maps native graph copy activities: all computational
kernels match eager exactly; the12 added graph memcpy kernels are enclosed
within one h/z gate each. Eager traces independently retain12 DtoD copies of
524288/2097152 bytes at z/h. All copies stay in the measured GPU activity budget;
strict-matcher failures are retained, not edited. This changes attribution only.
