# Training-log semantics and finite-budget interpretation

Source script: Analysis 020/01_audit_logs.py. Original generating code is
verified against each run manifest (336 relevant source references).
`adamw_gradient_norm_pre_clip` is torch clip_grad_norm_'s global L2 norm over
all trainable parameters with gradients, after microbatch accumulation and AMP
unscaling, before clipping. These single-GPU workers have no distributed shards.
For the chosen A0/A4-OL1/A7-OL1 cohort this is task-only; naive-L1 records instead
clip a combined objective and are not in this plot. `task_gradient_norm` in
OL1 is computed after clipping and is not substituted for the raw field.
Post-clipping norms are bounded near one. OL1 eligible-subset adaptive norms
have a different scope and are analyzed separately.

Chosen before examining outcomes: A0 and A4-OL1/A7-OL1 at kappa=.5, each at
14M/70M/410M; early steps1..142, late570..712. All nine curves retain712
boundaries, no gaps, overflows or skipped updates. Input tokens advance by
2,097,152 per boundary to1,493,172,224. Successful-update counters include the
first optimizer call with LR zero (no parameter displacement). Raw curves are
unsmoothed; the shaded final20% uses all143 records. OLS regressions below use
raw task loss against billion input tokens; they are descriptive, not a
statistical test over independent seeds. First/last20 means summarize this
same fixed late window, without choosing a flatter or steeper subwindow.

The loss is the mean of task cross-entropies over the same boundary's equally
sized microbatches, before its optimizer update. It is not validation loss.
Validation events occur at steps1 and712; there is no intermediate or late
validation trend to infer. Terminal canonical evaluation is separately retained.
Learning-rate schedules are plotted: peak .001/.001/.0003, final .0001/.0001/.00003.
Tokens per parameter are106.14/21.20/3.68.

| Condition | Late median pre-clip norm | Late slope (loss/B tokens) | First20 mean | Last20 mean | Late clipped /143 |
|---|---:|---:|---:|---:|---:|
| 14M:A0:None | 0.317152 | -0.141458 | 5.274998 | 5.237428 | 0 |
| 14M:A4-OL1:0.5 | 0.365141 | 0.009624 | 6.106561 | 6.107096 | 0 |
| 14M:A7-OL1:0.5 | 0.330344 | 0.020719 | 5.902472 | 5.905640 | 0 |
| 70M:A4-OL1:0.5 | 0.411961 | -0.009111 | 5.423592 | 5.419877 | 0 |
| 70M:A7-OL1:0.5 | 0.363469 | -0.066376 | 5.240618 | 5.221273 | 0 |
| 410M:A4-OL1:0.5 | 0.418478 | -0.131833 | 5.257526 | 5.222217 | 0 |
| 410M:A7-OL1:0.5 | 0.442942 | -0.105067 | 5.179639 | 5.151678 | 0 |
| 70M:A0:None | 0.295014 | -0.219363 | 4.050639 | 4.000268 | 0 |
| 410M:A0:None | 0.850118 | -0.444801 | 4.584551 | 4.473818 | 33 |

410M A0 continues improving late (slope -.4448 loss/B tokens), as do the
smaller A0s (-.1415 and -.2194). Its median pre-clip norm .8501 exceeds .3172
and .2950. The 410M sparse high-threshold curves also improve (-.1318/- .1051),
whereas 14M versions are approximately flat/slightly increasing in this window.
These records support a finite-budget hypothesis, not a causal attribution:
parameter scaling, stochastic gradients, recipe and learning rate differ.
No matched continuation was run; longer training need not restore any assumed
ordering or affect dense and sparse recipes equally.

Outside these complete cohorts, the first 14M A1-H train line is corrupted
(4,055 NUL characters); it is not repaired or imputed. Readable records total
38,447/38,448 and contain no skipped updates. All original files remain unchanged.
