# TrialAgentBench

A clinical trial compares treatments in participants to estimate their benefits
and harms. Those estimates inform whether a drug candidate should advance,
stop, or be studied again. A statistical analysis can run correctly while
estimating the wrong treatment effect or ignoring a problem in the data.
TrialAgentBench evaluates whether an AI agent recognises what the evidence
supports and uses that conclusion in the next development decision.

The repository contains the public benchmark harness and its independent
verifier:

- [`harness/`](harness/) presents and grades the TrialEval and TrialDev tasks.
- [`verification/`](verification/) independently reconstructs the supported
  analyses and decisions from the same evidence shown to the agent.

Benchmark archives are distributed through the
[TrialAgentBench dataset](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/).

## Benchmark tasks

### TrialEval: does the analysis change when the evidence changes?

TrialEval contains 100 independently generated clinical trials. Each trial is
presented in five forms, giving 500 analysis tasks. Every task defines the
participants to analyse, the treatments to compare, the outcome and follow-up
period, how to handle events such as treatment switching, and the treatment
effect to estimate. The agent must then:

1. identify the participants who belong in the analysis;
2. choose an analysis appropriate for the trial and its data;
3. estimate the treatment effect and its uncertainty; and
4. plan the next trial using that result.

The five forms show the same trial with or without a specified analysis, as
tables prepared for analysis or as raw participant records, and with one
version containing a known duplicate that must be removed. The treatment
question remains unchanged. Their comparison tests whether the agent responds
to the evidence rather than repeating a familiar default. The benchmark
accepts multiple correct answers: an analysis receives credit when it estimates
the requested treatment effect and is appropriate for the supplied data.

### TrialDev: does the conclusion support the next decision?

Drug development requires a sequence of decisions after each new analysis:
which trial to run, whether the evidence is strong enough to continue, and
whether safety or lack of benefit requires stopping. TrialDev tests whether an
agent carries its statistical conclusion through that sequence in two settings:

- **Development of one drug candidate:** 200 tasks follow a single candidate
  through evidence review, trial design, safety assessment, an early test of
  benefit, a later confirmatory trial, and stop-or-advance decisions. This is
  the single-asset setting.
- **Allocation across several drug candidates:** 96 tasks give the agent three
  candidates and a fixed development budget. The agent must choose a lead
  candidate, retain a reserve, and decide whether new evidence justifies
  stopping the lead or moving resources to the reserve. This is the portfolio
  reallocation setting.

The first setting tests whether the analysis remains coherent along one
development path. The second also tests whether the resulting decision remains
feasible when several candidates compete for the same resources.

## What the evidence establishes

Synthetic trials allow a feature that determines which analyses are supported,
such as confounding or loss to follow-up, to change while the treatment
question and the rest of the evidence remain fixed. This makes it possible to
test whether an agent notices the change and responds appropriately. The
comparison is credible only if the generated records preserve the relationships
required by the analysis and if an external researcher can inspect how the
evidence leads to the grade.

- **The path from evidence to grade is independently inspectable.** The
  verifier repeats the accepted analyses and supported development decisions
  from the same files supplied in each task. A researcher can therefore
  determine whether a grade follows from the analysis supported by the data and
  locate a disagreement in the choice of method, its assumptions, or its
  calculation.
- **The synthetic trials preserve the information that determines a valid
  analysis.** Synthetic datasets generated with the same sample sizes as the
  public PATENCY, TERECO, and HeadSOAR trials reproduce when events occurred,
  how follow-up ended, repeated measurements, ordered outcomes, and safety
  events. Matching each column separately would not be enough: attaching
  measurements to the wrong participant can preserve plausible values while
  corrupting correlations and adjusted treatment estimates. Across additional
  public datasets, keeping each participant's record intact reduced
  within-participant correlation error by 30% and bias in the adjusted
  treatment estimate by 58% compared with independently sampling each column.
- **The capability measured by the benchmark changes simulated development
  outcomes.** Across 7,200 programmes, the best action among those supported by
  both the estimated effect and its uncertainty had a mean probability of
  successful programme completion of 0.712. Selecting from the estimated
  effect alone achieved 0.656, while choosing by an arbitrary candidate label
  achieved 0.653. All three strategies were evaluated under the same budget
  limits.

The [verification guide](docs/VERIFICATION_GUIDE.md) links each claim to its
code, numerical result, and source data.

The complete methods, numerical results, and source records are available in:

- [Statistical evidence index and headline results](verification/validation_results/REPORT.md)
- [Detailed statistical evidence chapters](verification/validation_results/reports/)
- [`verification/validation_results/RESULTS.csv`](verification/validation_results/RESULTS.csv)
- [`verification/validation_results/SOURCES.md`](verification/validation_results/SOURCES.md)
- [TrialDev decision consequences](verification/validation_results/trialdev_v1/REPORT.md)

## Install and inspect

With Python 3.11 to 3.13 and [`uv`](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/TrialAgentBench/TrialAgentBench-harness.git
cd TrialAgentBench-harness
uv sync --all-packages --all-extras
uv run --package trial-agent-bench trialagentbench --help
uv run --package trialagentbench-validation trialagentbench-validate --help
```

The repository is organised around the path from task evidence to independent
reconstruction:

```text
.
├── harness/          tasks, execution, grading, and replay
├── verification/     independent analyses, validation results, and figures
├── docs/              verification and reproducibility guides
├── Makefile           common checks, tests, builds, and smoke tests
└── pyproject.toml     reproducible two-package uv workspace
```

Run the package checks with:

```bash
make install
make check
make test
make build
make smoke
```

`make smoke` installs both built packages into a fresh virtual environment and
checks their command-line interfaces. Data-specific reconstruction commands
require the matching benchmark archives and are described in the
[reproducibility guide](docs/REPRODUCIBILITY.md).

Live model execution uses the network-isolated Docker environment. Build and
test it separately with:

```bash
make -C harness test-executor
```

## Citation, licence, and contributions

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The code and
documentation are distributed under [CC BY-NC 4.0](LICENSE). See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for reproducible defect reports and focused
contributions.
