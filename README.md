# TrialAgentBench

An AI agent can produce reproducible trial estimates and still drive the wrong development
decision, either because the trial's default analysis fails on its data or because the agent
corrects a problem the data do not have. TrialAgentBench measures whether agents recover
the decision-relevant treatment effect from realistic trial records and commission evidence worth
its cost under a budget. It prices each error by the development decision it changes, in simulated
worlds whose true effects are known.

This repository holds the harness that runs agents on the benchmark, the independent verifier that
re-scores them, and the documentation. The task data are on
[Hugging Face](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench).

## What it measures and why

![The confirmatory trial each reported estimate drives at three interval widths, beside the posterior-expected regret of five distortions of an honest report.](docs/figures/decision_layer.svg)

*How an answer is priced: each reported estimate and interval drives a decision to stop or to run a
confirmatory trial of a chosen size (left), and every distortion of an honest report, from a shifted
estimate to the agent's own shrinkage towards the prior, costs expected value against it (right).*

| Benchmark | The agent's work | The score |
| --- | --- | --- |
| **TrialEval** | Analyse a completed randomised trial from its protocol and participant-level records, in Python or R | Expected US$ millions lost when the answer leads to a different development decision (stop, or confirm at a chosen size) from an analysis verified to recover the true effect (posterior-expected regret); numerical loss is secondary |
| **TrialDev** | Manage a drug-development programme (one simulated world): analyse observational evidence on candidate treatments, commission randomised studies under a budget and a six-year horizon, then stop or advance | Risk-adjusted development value, compared with stopping at once and with an oracle that knows every true effect |

The primary numerical loss is proper: its expected value is minimised by the median estimate and
central interval at the assigned confidence level, in the declared comparison units. Numerical
summaries display bounded recovery as `1 / (1 + aggregate mean loss)`, using the same weights and
completeness rule as mean loss. On the panel being compared, higher recovery means lower mean loss.
Expected recovery over truth draws is not proper and is not an optimisation objective.
Recovery intervals transform the loss limits in reverse order; standard errors use the delta method.

The reference answers are the true effects of the simulated worlds. A **credited analysis** is
shown to recover the effect, respects the trial's design and retains enough information to avoid
consequential loss of decision value. More than one analysis can be credited; each costs zero regret.

Some trials are **traps**: the default analysis fails on their data and changes the decision.
Others are **twins** (benign decoys): a detectable departure tempts the same correction, but the
default remains valid. The correction is unbiased there yet wastes information and costs decision
value. A **series** is one clinical question and trial design; a **cell** applies one assumption
**tier** (A1: default valid; A2: twin; A3: trap) to a series. Each trap is paired with a benign cell
of its series, usually a twin. Their documents are identical after masking identifiers, dates and
data-dependent times, so the protocol cannot tell them apart. An agent that always corrects loses
value in the twins.

Regret prices the decision an answer changes: stop or confirm, or the confirmatory trial's size.
An answer that drives the same decision as a credited analysis costs zero regret, even if its
numerical estimate differs.

The public split is for development and self-reported results: its evaluator files are public, and
its task counts show which series hold a trap and its counterpart. Leaderboard claims use the
withheld split, which the operator holds and in which a trap and its counterpart never share worlds.

Withheld evaluation is not yet open; the procedure will be announced on the site and dataset card.

The effect anchors calibrate the conditional mean of the credited z statistic at each effect multiplier, compiled to the stated Monte Carlo precision, so that drawn signal-to-noise ratios follow the series-truncated Cochrane mixture. This calibration does not establish the realised z distribution as the mixture plus unit-normal noise.

Scores describe performance in simulated populations. They do not measure clinical effectiveness,
patient-care competence or regulatory fitness. [METHODS.md](docs/METHODS.md) sets out the
construction, grading, construct-validity argument and limitations.

## Quickstart

With Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/) and Docker running, install the locked
environment and run the bundled clinical demonstration:

```bash
git clone --branch v1.0.1 https://github.com/TrialAgentBench/TrialAgentBench-harness
cd TrialAgentBench-harness
uv sync --locked
docker build --tag trialagentbench/executor:1.0.1 executor
uv run trialagentbench demo run --output results/demo
```

Installation and the first container build need network access. The demo then runs offline, with
no benchmark download, provider extra or credential. Open `results/demo/index.html` to inspect
estimates, intervals, replayed code and development consequences. It uses a scripted analyst;
its scores measure no model. Use a new output directory for a fresh run, or add `--resume`.

Continue with [RUNNING.md](docs/RUNNING.md) to download and verify the dataset and run the
three-task smoke campaign. The headline main selection is 852 tasks per model and repetition
(84 TrialEval, 768 TrialDev). The guide covers model configuration, spending limits, task selections,
resuming interrupted runs and results; [METHODS.md](docs/METHODS.md) explains the scores, and the
[verifier guide](verifier/README.md) explains independent reproduction.

## Connect your agent

Replace the fixed policy in
[`examples/clinical/run_external_provider.py`](examples/clinical/run_external_provider.py) with your
agent client. The harness keeps control of evidence release, study actions, analysis execution and
grading, and reference answers never reach the agent's messages or tools. Campaigns route models
through OpenRouter; direct runs also accept OpenAI-compatible endpoints, including local servers.
[RUNNING.md](docs/RUNNING.md#8-connect-your-own-agent) describes each route and the reporting
contract.

## Demonstrations

The [demonstrations](examples/README.md) rebuild the evidence behind the benchmark's claims with
ordinary statistical libraries: how the decision layer prices an answer and why distorting it does not
pay, the fixed-policy panel of both splits against which data-blind strategies are judged, and the
TrialDev programmes in which budget buys evidence and evidence buys value.

## Documentation

| Document | Reader and purpose |
| --- | --- |
| [docs/RUNNING.md](docs/RUNNING.md) | Operators: installation, download and verification, selections, configuration, execution and results |
| [docs/METHODS.md](docs/METHODS.md) | Reviewers and statisticians: targets, construction, grading, qualification, construct validity and limitations |
| [verifier/README.md](verifier/README.md) | Reproducers: verify a download and re-score a campaign without the harness |
| [examples/README.md](examples/README.md) | Anyone wanting to inspect the scoring rules and release evidence: runnable demonstrations |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributors |
| [CHANGELOG.md](CHANGELOG.md) | What this release contains |

## Citation and licence

```bibtex
@misc{segal2026trialagentbench,
  author = {Segal, Bradley and Bolton, William and Torr, Philip and Clifton, David and Clifton, Lei},
  title  = {TrialAgentBench: Evaluating AI Agents for Clinical-Trial Analysis and Long-Horizon Drug-Development Decisions},
  year   = {2026},
  url    = {https://github.com/TrialAgentBench/TrialAgentBench-harness}
}
```

The code and documentation are released under [Apache-2.0](LICENSE); the benchmark data on Hugging
Face under CC BY 4.0. [Citation metadata](CITATION.cff) ·
[Issues](https://github.com/TrialAgentBench/TrialAgentBench-harness/issues)

<!-- release-facts:start -->

Both TrialDev splits are qualified: 384 public programmes (3,072 tasks) and 64 withheld programmes (512 tasks), 0 failed programmes. The TrialDev evaluation runs every programme at its three budget rungs and at profit x0.5, x1 and x2 on shared recruitment seeds. Its gradient criterion asks, in every programme and profit scale, that the constrained-to-lean gain of the oracle and of the reference policy be resolved (one-sided Holm-adjusted p ≤ 0.025), that the reference policy beat stopping, readout ranking and the fixed first candidate (z ≥ 1.96), and that the oracle stop at day 0 in under half of the programmes. The reference policy was calibrated only on development worlds that share no world with either split, then frozen and evaluated once. On the public split the gain is resolved in 12 of 12 programme x profit-scale cells for the oracle and in 12 of 12 for the reference policy, which beats all three controls in 12 of 12. At profit x1 the oracle stops at day 0 in 21.6% [17.8, 26.0] of public programmes, inside the 8.8-36.0% band its candidate law implies.

| Programme | Profit | Gain resolved (oracle / reference) | Reference over stop, readout ranking, fixed first (z) | Oracle stops at day 0 |
| --- | --- | --- | --- | --- |
| Atopic dermatitis | x0.5 | yes / yes | 7.3, 8.4, 7.4 | 31.2% [22.9, 41.1] |
| Atopic dermatitis | x1 | yes / yes | 8.5, 8.3, 6.9 | 27.1% [19.2, 36.7] |
| Atopic dermatitis | x2 | yes / yes | 9.1, 8.2, 6.8 | 26.0% [18.3, 35.6] |
| COPD exacerbation | x0.5 | yes / yes | 4.5, 9.0, 10.3 | 24.0% [16.5, 33.4] |
| COPD exacerbation | x1 | yes / yes | 7.6, 9.5, 9.4 | 16.7% [10.5, 25.4] |
| COPD exacerbation | x2 | yes / yes | 8.9, 9.8, 9.2 | 11.5% [6.5, 19.4] |
| Migraine prevention | x0.5 | yes / yes | 7.2, 8.7, 7.1 | 29.2% [21.0, 38.9] |
| Migraine prevention | x1 | yes / yes | 8.4, 8.6, 6.7 | 21.9% [14.8, 31.1] |
| Migraine prevention | x2 | yes / yes | 9.2, 8.9, 6.8 | 16.7% [10.5, 25.4] |
| Rheumatoid arthritis | x0.5 | yes / yes | 6.5, 9.1, 7.8 | 37.5% [28.5, 47.5] |
| Rheumatoid arthritis | x1 | yes / yes | 7.3, 7.7, 6.7 | 20.8% [13.9, 30.0] |
| Rheumatoid arthritis | x2 | yes / yes | 8.0, 7.4, 6.3 | 16.7% [10.5, 25.4] |

The generous rung is non-binding: the oracle's realised lean-to-generous gain is exactly zero in 3 of 12 cells and its 95% interval includes zero in all 12; its planned gain is at most $7.6M. On 384 dedicated withheld qualification worlds the oracle stops at day 0 in 22.1% [18.3, 26.6] of programmes at profit x1, and the gradient criterion is met.

<!-- release-facts:end -->
