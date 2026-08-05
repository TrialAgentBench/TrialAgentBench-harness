# TrialAgentBench harness

Clinical-trial reasoning depends on more than producing a plausible estimate.
The analysis must answer the stated treatment question, remain supported by the
observed assumptions, and lead to a decision that reflects both the estimate
and its uncertainty. The TrialAgentBench harness makes each of those steps
explicit and independently gradeable.

It presents two related tasks:

- **TrialEval** tests whether an agent selects and executes a supported analysis
  of participant-level trial data.
- **TrialDev** tests whether the agent carries the resulting conclusion through
  trial design, safety assessment, and stop-or-advance decisions.

Benchmark archives are available from the
[TrialAgentBench dataset](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/).
The independent analyses are included in the
[verification package](../verification/).

## How TrialEval tests evidence-responsive analysis

TrialEval separates the statistical question from the method used to answer it.
Each item states which participants, treatments, outcome, follow-up period, and
treatment effect must be analysed, including how to handle events such as
treatment switching. The agent chooses an analysis, checks whether it is
appropriate for the supplied data, estimates the treatment effect and its
uncertainty, and plans the next trial. The benchmark accepts multiple correct
answers: an analysis receives credit when it estimates the requested effect
and is appropriate for the supplied data.

Three factors determine the evidence supplied:

- **Design (D)** changes the allocation, analysis unit, endpoint observation,
  or design adjustment required by the study.
- **Assumption (A)** changes whether a usual analysis remains supported. The
  sequence moves from compatible conditions through detectable failures to
  cases requiring an alternative analysis, bounded conclusion, qualified
  uncertainty statement, or abstention.
- **Context (C)** changes the analysis specification or data preparation while
  preserving the treatment question. The fifth context contains one known
  duplicate record that must be removed before analysis.

The dataset contains seven design profiles and 25 score-bearing regime cells.
Four independently generated base trials are created for each cell and each
base trial is presented in five contexts, giving 500 TrialEval items. Comparing
matched contexts shows whether the analysis changes because the evidence
changes, rather than because a different question was asked.

## How TrialDev tests evidence-to-decision coherence

TrialDev begins with evidence review and follows the consequences of the
analysis through later development decisions.

The **single-asset stream** follows one drug candidate through 50 development
trajectories evaluated under four objectives, giving 200 programme views. The
agent designs each reached trial, analyses efficacy and safety, and decides
whether the candidate should stop or advance.

The **portfolio-reallocation stream** asks the agent to manage three drug
candidates under a fixed development budget. Twelve evidence settings are
crossed with four objectives and two budgets, giving 96 programme views. The
agent selects a lead and reserve, then decides whether new evidence supports
continuing the lead, stopping it, or moving resources to the reserve.

Both streams require actions to be supported by the available evidence and its
uncertainty. The portfolio stream additionally tests whether the action remains
feasible when capital and trial capacity must be allocated between candidates.

## Why the data roles are separated

Each archive member has one role:

- `participant` contains the task material that may be shown to the agent;
- `evaluator` contains the records used for deterministic offline grading; and
- `verification` contains the inputs needed for independent reconstruction.

Only participant-role files enter the agent workspace. The TrialDev runner
performs this projection from the complete archive. This separation ensures
that the agent is graded against information it could legitimately derive from
the task, while a researcher can still reconstruct the grading result
independently.

## Install

The supported Python range is declared in `pyproject.toml`.

```bash
git clone https://github.com/TrialAgentBench/TrialAgentBench-harness.git
cd TrialAgentBench-harness
python -m venv .venv
. .venv/bin/activate
python -m pip install ./harness
trialagentbench --help
```

Offline grading, replay, and analysis require no provider credentials. Optional
provider adapters are installed with:

```bash
python -m pip install "./harness[providers]"
```

The exact API transport forms part of a run identity. All supported transports
use the same task evidence, local tools, structured submission contracts,
network-isolated executor, and offline grader.

## Choose the next document by task

- [Quickstart](docs/QUICKSTART.md) installs the harness and runs or grades a
  benchmark archive.
- [Benchmark and data](docs/BENCHMARK_AND_DATA.md) defines the statistical
  questions, matched task structure, and archive roles.
- [Scoring](docs/SCORING.md) explains how analysis routes and supported actions
  receive credit.
- [Reproducibility](docs/REPRODUCIBILITY.md) distinguishes deterministic
  replay from independent statistical reconstruction.
- [Contracts](docs/CONTRACTS.md) specifies the schemas, checksums, and paths
  used by each machine interface.
- [Experiments](docs/EXPERIMENTS.md) specifies the controlled assistance and
  interface comparisons supported by the harness.

The JSON schemas and command help define the machine interfaces. The
documentation explains why each interface is needed to preserve the
statistical question from task presentation through grading.

## Inspect the independent evidence

The [verification package](../verification/) independently repeats the
analyses and decisions used for grading from the files supplied in each task.
This makes the path from evidence to grade inspectable. It also tests whether
the synthetic trials preserve source-trial outcomes and participant
relationships, and whether controlled changes have the expected analytical
consequences.

The evidence is organised in:

- the [verification guide](../docs/VERIFICATION_GUIDE.md);
- the [integrated report](../verification/validation_results/REPORT.md); and
- the [complete numerical summary](../verification/validation_results/RESULTS.csv).

## Check the harness

```bash
python -m pytest
```

The tests cover strict data contracts, accepted and rejected analysis routes,
nonidentified cases, deterministic phase replay, and portfolio resource
constraints.

## Citation, licence, and contributions

Citation metadata is provided in [CITATION.cff](CITATION.cff). The harness is
distributed under [CC BY-NC 4.0](LICENSE). See
[`../CONTRIBUTING.md`](../CONTRIBUTING.md) for reproducible defect reports and
focused contributions.
