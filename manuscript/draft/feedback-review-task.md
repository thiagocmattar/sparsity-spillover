We are revising an ICLR paper on activation sparsification in Pythia. The original study compared 4-site vs. 7-site thresholding, with either no OL1 pressure or OL1 applied at all thresholded sites. We have now added an important missing control: **h-only OL1 pressure while keeping broader thresholding**. At 14M we now have full \(\kappa\in\{0,.01,.05,.1,.5\}\) sweeps for 4-site and 7-site thresholding under no pressure, h-only pressure, and all-site pressure. The new result is that **threshold scope and pressure scope should be treated as separate design axes**. At moderate thresholds, h-only pressure provides most of the favorable quality effect, while broad pressure adds sparsity at a larger quality cost. At \(\kappa=.5\), broad 7-site pressure becomes especially valuable: A7+h reaches 16.66% \(S_{\mathrm{model}}\) at loss 5.732, while A7+all reaches 27.48% at 5.829 (+10.82 pp sparsity for +0.097 loss). Holding h-only pressure fixed also gives a cleaner threshold-placement comparison: at \(\kappa=.5\), A7+h improves over A4+h by +6.44 pp \(S_{\mathrm{model}}\) for only +0.009 loss. An attractive moderate point is A7+h at \(\kappa=.05\): 10.13% sparsity at loss 5.195 versus dense loss 5.209, though all results remain single-seed. The manuscript should therefore move away from a simple “4-site vs. 7-site ±OL1” framing toward **threshold placement → pressure placement → activation structure → kernel-exploitable structure → latency**. The task is to produce precise, surgical manuscript changes—figures, tables, terminology, framing, analyses, claims, and limitations—without rewriting the paper yet. Do not claim h-only scaling yet; 70M h-only runs are pending, and changes made now should remain valid regardless of those results.

1. **Make pressure placement a first-class design variable in the title/thesis.** A good minimal title revision is **“Activation Sparsification in Transformers: Threshold Placement, Pressure Placement, and Sparse Execution.”** Your methodology already allows \(\mathcal N\neq\mathcal P\), but the current framing does not capitalize on it. 

2. **Introduce explicit pressure-scope notation everywhere.** Stop using bare “A4-OL1” and “A7-OL1” once h-only appears. I would use either \(T_4/P_0,T_4/P_h,T_4/P_4\) and \(T_7/P_0,T_7/P_h,T_7/P_7\), or the more readable `4-site`, `4-site + OL1(h)`, `4-site + OL1(all)`. Do the same for seven-site. The important thing is that “OL1” alone can no longer mean a unique intervention.

3. **Rewrite one sentence in the abstract.** Replace the generic claim that “combining thresholding with pressure yields the highest sparsity” with the sharper finding: **pressure placement changes the quality–sparsity trade-off; h-only pressure gives the favorable moderate-threshold regime, while broad seven-site pressure becomes useful for extreme sparsity.** Keep the 27.48% endpoint, but present it as the high-sparsity regime rather than the universal best recipe. The current abstract overemphasizes “more pressure + broader placement = better.” 

4. **Rewrite the Introduction’s practical question.** Change it from essentially “where should interventions be placed?” to: **“Where should zeros be exposed, and independently, where should sparsity pressure be applied?”** Then add the execution question: **“Which resulting zero structure can kernels exploit?”** This produces the three-axis paper:
   \[
   \text{threshold placement}\rightarrow\text{pressure placement}\rightarrow\text{execution realization}.
   \]

5. **Rewrite the contribution summary around three findings, not methods.** The three claims should be: (i) threshold and pressure placement are separable; (ii) under fixed h-only pressure, seven-site thresholding increases computational reach with very small additional quality cost; (iii) logical sparsity and executable sparsity differ, with MMA structure better explaining latency. The current introduction instead foregrounds pressure/thresholding as interventions and kernel development. 

6. **Update Table 1 to include the h-only recipes.** Add:
   \[
   T_4/P_h:\quad \mathcal N=\{a,m,h,z\},\ \mathcal P=\{h\}
   \]
   and
   \[
   T_7/P_h:\quad \mathcal N=\{a,m,h,z,q,k,v\},\ \mathcal P=\{h\}.
   \]
   Do not call them auxiliary or accidental in the main paper. They are legitimate interventions allowed by the method definition. 

7. **Update the experiment counts.** Assuming all ten h-only checkpoints use the same final-checkpoint protocol, your 14M cohort becomes **40 trained endpoints**, and the full study becomes **64 endpoints** rather than 54. Update every occurrence of “30 14M conditions,” “54 trained endpoints,” etc. The existing protocol currently documents one seed and the original 30-condition 14M cohort. 

8. **Replace Figure 1 with the new pressure-scope overview, but simplify its encoding.** Keep the six multisite curves:
   - A4 \(P_0,P_h,P_4\)
   - A7 \(P_0,P_h,P_7\)

   Keep A0/ReLU/post-hoc in muted background. Encode **color = threshold scope** and **marker/fill/line = pressure scope**. Remove run IDs. The figure should visually answer “what happens as pressure scope expands?” rather than function as a catalog.

9. **Rewrite Section 4.1 around operating regimes.** The central 14M result should be:
   - **moderate \(\kappa\le0.1\):** h-only pressure gives the large quality improvement;
   - **high \(\kappa=.5\):** broad seven-site pressure buys a large additional sparsity increment.

   One particularly useful operating point is \(T_7/P_h,\kappa=.05\):
   \[
   S_{\rm model}=10.126\%,\qquad L=5.19496,
   \]
   versus dense \(L=5.2086\). Phrase this as **comparable loss at the evaluated seed**, not evidence that sparsification improves generalization, because you still have one training seed. The one-seed limitation is explicit in the protocol. 

10. **Completely repurpose Figure 3.** The old Figure 3 is “marginal effect of adding OL1.” Replace it with:
   \[
   P_0\rightarrow P_h\rightarrow P_{\rm all}
   \]
   separately for A4 and A7. Show both \(\Delta L\) and \(\Delta S_{\rm model}\). This is now the most important causal-design figure in the paper.

11. **Report these two numerical contrasts prominently.** First, threshold placement at fixed pressure:
   \[
   T_7/P_h-T_4/P_h,\quad \kappa=.5:
   \]
   **+6.44 pp \(S_{\rm model}\) for +0.0094 loss.**

   Second, pressure expansion at fixed seven-site thresholding:
   \[
   T_7/P_7-T_7/P_h,\quad \kappa=.5:
   \]
   **+10.82 pp \(S_{\rm model}\) for +0.0974 loss.**

   These two contrasts separate **broadening threshold reach** from **broadening pressure scope** much better than the original A4-OL1 vs A7-OL1 comparison.

12. **Rename Section 4.3.** Replace “Broader threshold placement without pressure” with something like **“Broader threshold placement under matched pressure scope.”** Analyze both:
   \[
   T_7/P_0-T_4/P_0
   \]
   and
   \[
   T_7/P_h-T_4/P_h.
   \]
   The second contrast is especially valuable because pressure is held fixed at \(h\). Your existing section only provides the pressure-free contrast. 

13. **Replace Table 2 with a matched-threshold-scope table.** For every \(\kappa\), give two rows/columns:
   - A7−A4 with \(P_0\)
   - A7−A4 with \(P_h\)

   Report \(\Delta L\) and \(\Delta S_{\rm model}\). This will make the near-zero quality cost of broader thresholding under h-only pressure immediately visible.

14. **Do not say h-only proves that Q/K/V pressure is unnecessary.** You still lack:
   \[
   T_7/P_4,
   \]
   so \(P_h\rightarrow P_7\) simultaneously adds pressure at \(a,m,z,q,k,v\). You **cannot isolate the contribution of Q/K/V pressure specifically**. State this once in Section 4.2 and again in limitations.

15. **Rebuild Figure 4 if the h-only OL1 geometry is available.** Compare:
   \[
   P_h,\quad P_4,\quad P_7
   \]
   on pressure/task norm ratio, opposing component, and cap-active fraction. The current Figure 4 already shows a huge four-vs-seven-site optimization-regime difference.  If h-only logs do not exist, **do not infer its geometry from final performance**; keep Figure 4 as-is and explicitly call the h-only mechanism unresolved.

16. **Redo Figure 5 using pressure-scope contrasts rather than only A4-all vs A7-all.** At minimum analyze:
   \[
   T_4/P_0,\ T_4/P_h,\ T_4/P_4
   \]
   and
   \[
   T_7/P_0,\ T_7/P_h,\ T_7/P_7
   \]
   at \(\kappa=.5\). Ideally add \(.05\) in the appendix because it represents the quality-preserving regime. Unpool \(m\) and \(h\); the current pooled FFN metric weights them 20:80 and hides their different behavior. 

17. **Change the language around “spillover.”** Use **“nonlocal response”** consistently unless the matched \(P_0\rightarrow P_h\) comparison directly establishes the relevant change. The current manuscript already uses the safer language and notes that the old comparison did not isolate pressure versus thresholding.  The h-only controls now let you make some of those statements more specific.

18. **Leave Figure 6 mostly untouched until 70M returns, but rename the section now.** Change “Transfer Across Model Sizes” to **“Cross-size recurrence of broad-pressure recipes.”** Your current evidence only establishes recurrence of the complete A4/P4 vs A7/P7 ordering; the larger cohorts lack the matched pressure controls.  Do not pre-write a h-only scaling conclusion.

19. **Prepare Figure 6 so the 70M result can slot in without redesign.** Use pressure scope as marker style. When the new models arrive, you should be able to add \(T_4/P_h\) and/or \(T_7/P_h\) as extra points at \(\kappa=.05,.5\), rather than inventing a new figure.

20. **Add all ten h-only checkpoints to the runtime evaluation if feasible.** The appendix explicitly says the five historical A4+h checkpoints were excluded from the 30-model kernel cohort because they did not implement the intended four-site pressure objective.  That rationale is obsolete now: h-only pressure is scientifically relevant. If they qualify, make the runtime cohort 40 models.

21. **Change Figure 7’s main deployment plot to validation loss vs absolute latency.** Put native-relative speedup in a secondary panel/appendix. This matters even more now because the h-only models could be Pareto-efficient in quality–latency space. Your existing appendix already demonstrates that native-relative speedup can reverse the absolute-latency ranking. 

22. **Keep the MMA-bypass result prominent.** The new conceptual chain should read:
   \[
   \text{pressure scope}
   \rightarrow
   \text{activation structure}
   \rightarrow
   \text{MMA-bypass structure}
   \rightarrow
   \text{latency}.
   \]
   You already have strong descriptive evidence that projection MMA bypass predicts projection-skipping gain much better than scalar sparsity does. 

23. **Rewrite the researcher recommendation.** The paper can now give an actual rule of thumb, carefully scoped to this study:
   **“Use threshold placement to expose computational reach; start pressure narrowly at the FFN hidden site when quality preservation matters, and broaden pressure only when the target operating point requires extreme sparsity.”**
   Add “in our 14M sweep” until the 70M result arrives.

24. **Rewrite the conclusion around separation, not maximum sparsity.** The first sentence should no longer be “pressure, thresholds and site placement need to be evaluated together.” It should communicate:
   \[
   \boxed{\text{where zeros are exposed need not equal where pressure is applied}}
   \]
   followed by the execution distinction. The current conclusion already argues that placement and execution structure matter; this new result gives it a much sharper organizing principle. 

25. **Add one explicit limitation paragraph about identification.** State that the h-only comparison isolates pressure scope more cleanly than the original recipes, but:
   - each condition still has one training seed;
   - \(T_7/P_4\) is absent, so Q/K/V pressure cannot be isolated;
   - h-only scaling is untested until the new 70M checkpoints are available;
   - larger-scale conclusions presently concern only complete broad-pressure recipes.

26. **Update Appendix Table 5 and contrast tables.** If all new endpoints pass the same evaluation protocol, rename it **“All 64 trained endpoints.”** Add the ten h-only rows. Replace/augment the current 14M contrast table with the useful decompositions:
   \[
   P_h-P_0,\quad
   P_{\rm all}-P_h,\quad
   T_7/P_h-T_4/P_h.
   \]
   The current Table 6 only contains the old pressure-on/off contrasts. 

27. **Globally replace “post-hoc clipping” with “post-hoc thresholding” unless you need the historical baseline name.** You zero values below a threshold rather than clipping their magnitude. This will align the terminology with your training-time thresholding definitions.

28. **Cut material that is now competing with the stronger story.** In particular, shrink the coding-agent/kernel-search narrative in the main text to one or two sentences and leave search-history details in the appendix. The scientific result is now intervention placement → representation → execution; the development process should not occupy equal narrative weight.

29. **Do not add the 70M claim yet.** Leave yourself a clearly marked source-level insertion point after the 14M pressure-placement result. When 70M arrives, the question is simple: **does the \(P_h\) regime recur?** If yes, promote it into Figure 6/abstract. If no, that becomes a useful scale-dependence result rather than forcing another reframing.

30. **Use the next few hours on three evaluation-only analyses while 70M runs:** site-by-layer sparsity for the six multisite families, h-only K050 timing, and h-only OL1 geometry if logs/checkpoints permit. Those three analyses determine whether the new result is merely a better frontier point or the mechanistic center of the paper.

The **single biggest structural change** is: stop organizing the paper around “A4 versus A7 +/− OL1.” Organize it around **threshold scope \(T\)** and **pressure scope \(P\)**. Almost every existing experiment then becomes easier to interpret, and the new h-only series stops looking like an extra ablation and becomes the missing axis that makes the original study legible.