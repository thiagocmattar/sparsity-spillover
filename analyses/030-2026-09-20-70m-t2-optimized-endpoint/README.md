# Later 70M T2/Ph optimized endpoint

21 September presentation revision: remove the extra diamond overlays and
"Later session" legend entry from Figure 3, and simplify its caption. All 27
coordinates and measurement provenance are unchanged. Only the figure PDF and
caption source are updated; the full manuscript is not recompiled for this edit.

Approved follow-up to Analysis028: add Run046's T2/Ph kappa=.5 checkpoint
after its Run047 optimized-kernel measurement qualifies. Retain all 26 original
Run045 coordinates. The later session repeats Base and kappa=.1 as references;
report those measurements separately, without rescaling or pooling sessions.
Canonical quality and logical sparsity are unchanged. No new training occurs.

`01_integrate.py` joins verified Run047 evidence, writes the two affected tables
and exports the complete 84-condition data. `02_plot.py` retains the earlier
figure's style and adds the later endpoint. `03_install_verify.py` installs
only these artifacts and checks their source and manuscript copies.

Status: verified. All nine Run047 processes qualify; all 217 returned files
pass hash/size checks and local reduction matches the remote result exactly.
The added point is 1.214368 ms, loss 4.838034 and sparsity 15.426970%, with
1.363778x speedup relative to its same-session native Base. The Pod is deleted;
estimated total cost is below USD0.37, within the approved USD2 cap.

All 83 other endpoint records, historical clipping, paired-pressure data and
14M figure membership remain unchanged. The figure now has 27 points; complete
results still contain 84 conditions (45/27/12). The two affected tables and
manuscript prose identify the later session. See [the observation and
caption](observations/001-later-70m-endpoint.md), `data/verification.json` and
`data/build-verification.json` for evidence and final PDF checks.

The checked build had 22 pages, no unresolved references or overfull boxes,
and two underfull-box warnings. Analysis031's later verified 21-page build
retains these results and adds the approved three-panel introduction figure;
it is now installed as `manuscript/draft/main.pdf` after the user closed the
viewer. The original Analysis030 PDF was compiled from a fixed
snapshot of the latest saved sources while the author continued editing the
introduction; only its 26-to-27 count change is staged by this task.
