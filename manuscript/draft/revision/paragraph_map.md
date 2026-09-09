# Frozen paragraph-anchor map

This map preserves the original runbook anchors used for the first rewrite. The
author-approved condensation merges those paragraphs; B21-B23 in `change_log.md`
and the current section files record the final organization. Scientific claims
retain their source boundaries in `claim_ledger.md` and TeX evidence comments.


Source: Git `1e14471bd4763dbfef84a162d4429021d2f7e2e6`. Line numbers locate the original TeX; anchors remain authoritative after reflow. Paragraphs continuing around a figure/page are mapped to their complete source paragraph. The JSON retains the original text.

| Task | File:lines | Heading | Opening anchor / action |
|---|---|---|---|
| RW-31-01 | `methodology.tex:9-16` | Methodology / Sparsification interventions | Q-Sparse selects the top-$k$ activation magnitudes~\citep{wang2024qsparse}.; edit anchored paragraph |
| RW-31-02 | `methodology.tex:18-21` | Methodology / Sparsification interventions | A sparsification intervention specifies where to set activations to zero; edit anchored paragraph |
| RW-31-03 | `methodology.tex:23-30` | Methodology / Sparsification interventions | We use ReLU and two thresholding nonlinearities:; edit anchored paragraph |
| RW-31-04 | `methodology.tex:32-35` | Methodology / Sparsification interventions | The threshold is fixed within each training condition. Our hypothesis is; edit anchored paragraph |
| RW-31-05 | `methodology.tex:37-48` | Methodology / Sparsification interventions | Pressure uses $\mathcal{L}_1$: we take the mean absolute activation in each; edit anchored paragraph |
| RW-32-01 | `methodology.tex:53-62` | Sparsification interventions / Sparsity and architectural reach | \emph{Activation sparsity} is the fraction of exactly zero entries at a named site.; edit anchored paragraph |
| RW-32-02 | `methodology.tex:64-71` | Sparsification interventions / Sparsity and architectural reach | The \emph{model-wide sparsity ceiling},; edit anchored paragraph |
| RW-40-01 | `experimental-study.tex:6-12` | Experimental study | We pretrain randomly initialized Pythia-14M, 70M, and 410M; edit anchored paragraph |
| RW-40-02 | `experimental-study.tex:35-41` | Experimental study | Within each size, matched contrasts share initialization, data order, and; edit anchored paragraph |
| RW-40-03 | `experimental-study.tex:43-51` | Experimental study | The cohort contains 30 conditions at 14M and 12 at each larger size.; edit anchored paragraph |
| RW-40-04 | `experimental-study.tex:53-59` | Experimental study | For the evaluation-only reference, we apply uniform magnitude-clipping targets; edit anchored paragraph |
| RW-41-01 | `training-results.tex:12-19` | Quality--sparsity trade-offs | Local pressure and broader nonlinearities offer different quality--sparsity trade-offs; edit anchored paragraph |
| RW-41-02 | `training-results.tex:21-29` | Quality--sparsity trade-offs | Broader interventions reach much greater sparsity. A7-OL1 at $\kappa=0.5$; edit anchored paragraph |
| RW-41-03 | `training-results.tex:21-29` | Quality--sparsity trade-offs | Broader interventions reach much greater sparsity. A7-OL1 at $\kappa=0.5$; new insertion after anchor |
| RW-42-01 | `training-results.tex:53-59` | Quality--sparsity trade-offs / Paired intervention effects | Pressure's additional value depends on the threshold and target set; edit anchored paragraph |
| RW-42-02 | `training-results.tex:61-67` | Quality--sparsity trade-offs / Paired intervention effects | For A4, adding OL1 improves both loss and sparsity at $\kappa=0$ and 0.01.; edit anchored paragraph |
| RW-42-03 | `training-results.tex:69-72` | Quality--sparsity trade-offs / Paired intervention effects | Pressure therefore needs to be evaluated at its actual threshold and sites.; edit anchored paragraph |
| RW-43-01 | `training-results.tex:94-102` | Paired intervention effects / Reshaping activation distributions | More FFN zeros need not mean more model-wide sparsity; edit anchored paragraph |
| RW-43-02 | `training-results.tex:104-109` | Paired intervention effects / Reshaping activation distributions | A4 does not directly threshold or pressure $q,k,v$, yet their distributions; edit anchored paragraph |
| RW-43-03 | `training-results.tex:111-117` | Paired intervention effects / Reshaping activation distributions | The operation counts show why this distinction matters model-wide. At the; edit anchored paragraph |
| RW-43-04 | `training-results.tex:111-117` | Paired intervention effects / Reshaping activation distributions | The operation counts show why this distinction matters model-wide. At the; new insertion after anchor |
| RW-44-01 | `training-results.tex:140-146` | Reshaping activation distributions / Transfer across model sizes | The high-threshold A7-OL1 advantage persists at all three model sizes; edit anchored paragraph |
| RW-44-02 | `training-results.tex:148-155` | Reshaping activation distributions / Transfer across model sizes | Raw sparsity must also be read against the changing workload. The A7 sparsity; edit anchored paragraph |
| RW-44-03 | `training-results.tex:157-161` | Reshaping activation distributions / Transfer across model sizes | This is transfer of a complete-recipe comparison. The larger cohorts lack; edit anchored paragraph |
| RW-45-01 | `kernel-autoresearch.tex:10-16` | Realizing sparsity with specialized inference kernels | Realizing activation sparsity requires kernels matched to the model's shapes,; edit anchored paragraph |
| RW-45-02 | `kernel-autoresearch.tex:18-26` | Realizing sparsity with specialized inference kernels | To make this specialization practical within an architecture study, we use; edit anchored paragraph |
| RW-45-03 | `kernel-autoresearch.tex:28-36` | Realizing sparsity with specialized inference kernels | The matched retrospective reaches $1.6024\times$ by iteration 11 and; edit anchored paragraph |
| RW-45-04 | `kernel-autoresearch.tex:38-49` | Realizing sparsity with specialized inference kernels | Ablations distinguish fusion from the contribution of sparse paths.; edit anchored paragraph |
| RW-45-05 | `kernel-autoresearch.tex:28-36` | Realizing sparsity with specialized inference kernels | The matched retrospective reaches $1.6024\times$ by iteration 11 and; replace/supplement anchor |
| RW-50-01 | `conclusion.tex:6-14` | Discussion and conclusion | Pressure, thresholds and site placement need to be evaluated together.; edit anchored paragraph |
| RW-50-02 | `conclusion.tex:16-23` | Discussion and conclusion | The interpretation is supported by a connected pattern across experiments.; edit anchored paragraph |
| RW-50-03 | `conclusion.tex:25-31` | Discussion and conclusion | Product accounting and kernel timing answer different questions. Counting; edit anchored paragraph |
| RW-50-04 | `conclusion.tex:33-38` | Discussion and conclusion | These conclusions concern MiniPile pretraining at a fixed token budget,; edit anchored paragraph |
| RW-21-01 | `related-work.tex:9-17` | Related Work / Sparsification interventions | Post-hoc thresholding removes activations that a trained model can tolerate; edit anchored paragraph |
| RW-21-02 | `related-work.tex:19-30` | Related Work / Sparsification interventions | Training-time interventions allow the representation to adapt to sparse; edit anchored paragraph |
| RW-21-03 | `related-work.tex:32-42` | Related Work / Sparsification interventions | Extending sparsification beyond FFNs makes the choice of sites explicit.; edit anchored paragraph |
| RW-22-01 | `related-work.tex:47-60` | Sparsification interventions / Inference speedup | Converting activation sparsity into speedup requires an execution strategy; edit anchored paragraph |
| RW-22-02 | `related-work.tex:62-74` | Sparsification interventions / Inference speedup | The execution strategy also depends on shape and sparsity structure.; edit anchored paragraph |
| RW-10-01 | `introduction.tex:19-19` | Introduction | Activation sparsity can improve language-model inference efficiency when specialized kernels skip computation and memory access associated w; edit anchored paragraph |
| RW-10-02 | `introduction.tex:35-35` | Introduction | Existing approaches induce activation sparsity in different ways. TEAL applies post-hoc thresholding to pretrained models, ProSparse uses L1; edit anchored paragraph |
| RW-10-03 | `introduction.tex:37-37` | Introduction | In this work, we study three sparsification interventions: (i) \emph{pressure}, an optimization regularizer that forces selected activations; edit anchored paragraph |
| RW-10-04 | `introduction.tex:64-64` | Introduction | We pretrain Pythia-family models from scratch at multiple scales to study sparsification under controlled conditions~\citep{biderman2023pyth; edit anchored paragraph |
| RW-10-05 | `introduction.tex:66-66` | Introduction | Finally, we test whether specialized kernels can translate these sparsity gains into faster inference using GPT-6-Astra. Across 42 eligible ; edit anchored paragraph |
| RW-ABS-01 | `abstract.tex:23-23` | Abstract | Activation sparsity can accelerate language-model inference when specialized kernels skip computation associated with zero activations. Exis; edit anchored paragraph |
| RW-TITLE-01 | `main.tex:13-32` | Title | \title{Activation Sparsification in Transformers:\\; edit anchored paragraph |
