# TrialAgentBench

TrialAgentBench evaluates AI agents inside a clinically grounded simulated research world. Agents must interpret clinical records, analyse treatment benefit and harm, and decide what further research is worth its cost. The world defines each question's target independently of the method an agent chooses, allowing different analyses and research strategies to be compared.

An agent receives a clinical brief, protocol and patient records from the fictional Ariadne Clinical Research Collaborative. It inspects the evidence, runs Python or R, and submits estimates, uncertainty intervals and reproducible code. In a development programme, it can design a study, preview its cost, recruit patients and acquire a readout. New evidence changes what the agent knows; studies consume funding and calendar time. The agent must decide whether to revise its analysis, buy more evidence, advance a treatment or stop.

| Task | Work assigned to the agent | Evaluation |
| --- | --- | --- |
| **TrialEval** | Analyse a completed trial from its clinical question, protocol and patient records | Decision regret (US$ millions) of the stop-or-confirm decision each estimate and interval drives, against the closest credited analysis; numerical loss secondary |
| **TrialDev** | Analyse initial evidence, commission randomized studies, and stop or advance a treatment | Analysis quality and development value under stated benefit, harm, cost and time assumptions |

**Available now:** the qualified TrialEval and TrialDev releases on [Hugging Face](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench), scored agent evaluation through this harness, an offline demonstration and scripts to reproduce the analyses. The teaching cases' example answers are public; use them to explore the environment and connect your agent.

## Try the research environment

With Python 3.12, Docker and [uv](https://docs.astral.sh/uv/), run from this checkout:

```bash
uv sync
docker build --tag trialagentbench/executor:1.1.1 executor
uv run trialagentbench demo run --output results/demo
```

Open `results/demo/index.html`. A scripted analyst works through ACRC clinical documents and records, saves and submits a kidney-trial analysis, then previews and runs a sixty-person safety study in a smaller MERIDIAN fixture before stopping. The report connects native tool actions to returned evidence, estimates, spending and development value. It needs no model API key; its scripted actions are not model performance results.

## Reproduce a clinical calculation

From the public source checkout, refit the colorectal adjudication example with ordinary statistical libraries. It needs neither Docker nor a model API key, but it reads `recovery-records.zip`, which the checkout does not contain. Download it from the dataset's [`scientific-evidence/`](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/tree/main/scientific-evidence) directory into `verifier/examples/scientific/` and check it there with `sha256sum -c --ignore-missing SHA256SUMS`.

```bash
uv sync --extra analysis
uv run python verifier/examples/scientific/review_adjudication.py --output results/colorectal-review
```

Open `results/colorectal-review/adjudication-worked.svg` for the estimates, `adjudication-grading.svg` for their grades, and `recovery-summary.csv` for all 32 paired trials. The [clinical guide](docs/GUIDE.md) connects the clinical world, research actions and consequences of interpreting the evidence. The [scientific walkthrough](verifier/examples/scientific/README.md) reproduces the figures, trial structures and research costs.

## Connect your agent

Replace the fixed controller in [the external-provider example](examples/clinical/run_external_provider.py) with your agent client. The harness continues to control evidence, study actions, execution and grading. Public examples let you test that integration before a scored evaluation.

The [run guide](docs/QUICKSTART.md) covers the task-data release, model configuration, execution limits, resuming campaigns and external agents. The [benchmark reference](docs/BENCHMARK_AND_DATA.md) specifies case construction, tools and scoring. Analysis accuracy and development value are separate results; performance in the simulated tasks does not establish clinical deployment readiness.

The [independent verifier](verifier/README.md) replays analyses and recalculates scores without importing the harness. See [CONTRIBUTING](CONTRIBUTING.md) for development and the [changelog](CHANGELOG.md) for release changes.

## Citation

Bradley Segal, William Bolton, Philip Torr, David Clifton and Lei Clifton.
*TrialAgentBench: Evaluating AI Agents for Clinical-Trial Analysis and Long-Horizon
Drug-Development Decisions.*

```bibtex
@misc{segal_trialagentbench,
  author = {Segal, Bradley and Bolton, William and Torr, Philip and Clifton, David and Clifton, Lei},
  title = {TrialAgentBench: Evaluating AI Agents for Clinical-Trial Analysis and Long-Horizon Drug-Development Decisions},
  url = {https://github.com/TrialAgentBench/TrialAgentBench-harness}
}
```

The code and documentation in this repository are released under [Apache-2.0](LICENSE); the benchmark data on
Hugging Face are released under CC BY 4.0. [Citation metadata](CITATION.cff) ·
[Issues](https://github.com/TrialAgentBench/TrialAgentBench-harness/issues)
