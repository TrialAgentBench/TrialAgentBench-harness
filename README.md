# TrialAgentBench

An AI agent can analyse a clinical trial competently by every surface check and still drive the
wrong development decision, either because the trial's default analysis fails on its data or because
the agent corrects a problem the data do not have. TrialAgentBench measures whether agents recover
the decision-relevant treatment effect from realistic trial records and commission evidence worth
its cost under a budget. It prices each error by the development decision it changes, in simulated
worlds whose true effects are known.

This repository holds the harness that runs agents on the benchmark, the independent verifier that
re-scores them, and the documentation. The task data are on
[Hugging Face](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench).

## What it measures and why

![Routine-record and adjudication-corrected analyses of the four public TrialEval worlds in which routine records miss more heart-failure events on the active arm, compared with the true effect, with the stop-or-confirm decision and regret each answer drives.](docs/figures/trap_decision.svg)

*A trap trial: routine records miss more heart-failure events on the active treatment. In four
simulated trials, the usual analysis and the adjudication-corrected analysis are compared with the true
effect, with the stop-or-confirm decision each leads to and what that decision costs (TE-S06-A3).*

| Benchmark | The agent's work | The score |
| --- | --- | --- |
| **TrialEval** | Analyse a completed randomised trial from its protocol and participant-level records, in Python or R | Expected US$ millions lost when the answer leads to a different stop-or-confirm decision from an analysis verified to recover the true effect (posterior-expected regret); numerical loss is secondary |
| **TrialDev** | Analyse observational evidence on several candidate treatments, commission randomised studies under a budget and a six-year horizon, then stop or advance | Risk-adjusted development value, compared with stopping at once and with an oracle that knows every true effect |

Three design choices make a score mean what it claims. The reference answers are the true effects
of the simulated worlds, so no method is privileged: the benchmark accepts (credits) every analysis
shown to recover the true effect while respecting the trial's design, and a credited analysis costs
nothing. Some
trials are **traps**: the usual analysis fails on their data and changes the decision. Each trap
has a **twin** whose documents are identical but whose data do not need the trap's fix; there the
fix is still unbiased but wastes information and changes decisions. Because the documents are
identical, reading the protocol cannot tell trap from twin, and an agent that always applies the
fix loses in the twin. Finally, each error is priced by the decision it changes, so a numerically
close answer that flips a confirmatory decision is penalised, and an answer that leads to the same
decision as a credited analysis is not.

Scores describe performance in simulated populations. They do not measure clinical effectiveness,
patient-care competence or regulatory fitness. [METHODS.md](docs/METHODS.md) sets out the
construction, grading, construct-validity argument and limitations.

## Quickstart

With Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/) and Docker running, download the
release, verify it and run the three-task smoke campaign, which uses scripted controls and no model:

```bash
git clone https://github.com/TrialAgentBench/TrialAgentBench-harness
cd TrialAgentBench-harness
uv sync
uvx --from huggingface_hub hf download TrialAgentBench/TrialAgentBench --repo-type dataset --revision v1.0.0 --include "SHA256SUMS" --include "full/*" --include "main/*" --exclude "full/shards/*" --local-dir benchmark
(cd benchmark && sha256sum --check --ignore-missing SHA256SUMS)
export TRIALAGENTBENCH_CACHE_DIR="$PWD/cache" TRIALAGENTBENCH_ASSET_BASE_URL=https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/resolve/v1.0.0/full
uv run trialagentbench campaign preview --config examples/smoke.json
uv run trialagentbench campaign prepare --config examples/smoke.json --campaign results/smoke
uv run trialagentbench campaign run --campaign results/smoke
uv run trialagentbench campaign finish --campaign results/smoke
```

Preparation builds the analysis container on first use and fetches only the shards the selected
tasks need. Open the printed `results/smoke/index.html`. The smoke scores check the installation;
they measure no model. The headline main selection is 852 tasks per model (84 TrialEval, 768
TrialDev); [RUNNING.md](docs/RUNNING.md) covers model configuration, spending limits, the selections and the
withheld split, resuming, recovery and results.

## Connect your agent

Replace the fixed policy in
[`examples/clinical/run_external_provider.py`](examples/clinical/run_external_provider.py) with your
agent client. The harness keeps control of evidence release, study actions, analysis execution and
grading, and reference answers never reach the agent's messages or tools. Campaigns route models
through OpenRouter; direct runs also accept OpenAI-compatible endpoints, including local servers.
[RUNNING.md](docs/RUNNING.md#8-connect-your-own-agent) describes each route and the reporting
contract.

## Demonstrations

The [demonstrations](examples/README.md) rebuild the evidence behind the benchmark's claims from the
public release with ordinary statistical libraries: a confident wrong decision in a trap, the cost of
correcting anyway in its twin, the fixed-policy panel in which no data-blind strategy passes, and a
TrialDev programme in which budget buys evidence and evidence buys value.

## Documentation

| Document | Reader and purpose |
| --- | --- |
| [docs/METHODS.md](docs/METHODS.md) | Reviewers and statisticians: targets, construction, grading, qualification, construct validity and limitations |
| [docs/RUNNING.md](docs/RUNNING.md) | Operators: installation, download and verification, selections, configuration, execution and results |
| [examples/README.md](examples/README.md) | Anyone wanting to see why the benchmark matters: runnable demonstrations |
| [verifier/README.md](verifier/README.md) | Reproducers: verify a download and re-score a campaign without the harness |
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
