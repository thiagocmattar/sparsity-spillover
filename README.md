# Activation Sparsity Research

Lean guided-research repository for studying activation pressure, exact-zero
gates, logical product opportunities, and quality trade-offs in randomly
initialized Pythia pretraining.

The repository includes the living paper draft under `manuscript/`. The draft
sharpens the research question, terminology, formal definitions, and intended
contributions, while executable contracts and empirical evidence retain their
separate authorities described in `research/MANUSCRIPT.md`.

The user chooses the research direction. The agent turns a plain-language
request into a precise experiment proposal, asks for design confirmation,
implements one numbered run folder, reports tests/ETC/location/cost, asks for
launch confirmation, executes, retrieves artifacts, terminates cloud resources,
and waits for instructions before analysis.

## First use

1. Copy everything in this directory, including `.gitignore`, into the root of
   a fresh repository. No second manuscript copy is required.
2. Start Codex in that repository and provide the setup-verification request.
3. Use an existing isolated environment or create `.venv`, then install the
   declared verification dependencies with `python -m pip install -e ".[dev]"`.
4. Run `python -m pytest`.
5. Read `AGENTS.md`, `research/INDEX.md`, `research/DEFINITIONS.md`,
   `research/DATA.md`, and `research/MANUSCRIPT.md`.
6. Read the current introduction/methodology under `manuscript/` for work tied
   to the paper's direction.
7. Describe the first experiment in plain language. The template in
   `runs/000-template/` is copied only after the design is confirmed.

The end-to-end human/agent sequence is in `research/WORKFLOW.md`; cloud training
details are in `research/RUNPOD.md`.

## Structure

```text
AGENTS.md
research/                  compact shared knowledge and methodology
manuscript/                living paper framing, formal draft, and future TeX
runs/NNN-YYYY-MM-DD-slug/  one approved experiment, including its plots
analyses/NNN-YYYY-MM-DD-slug/ cross-run analysis, including its plots
src/sparsity_research/     code genuinely shared by at least two runs
tools/                     small operational helpers
data/ and cache/           local, ignored
```

Completed runs are records, not libraries. Promote code into `src/` only when a
second run needs the same behavior.

The manuscript is neither a frozen plan nor an evidence store. It guides terms
and questions; run artifacts and approved findings support result claims.
