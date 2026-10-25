# Work through a clinical research episode

The sponsor, Ariadne Clinical Research Collaborative, and all simulated trial and programme names
are fictional. The collaborative asks an analyst to estimate
a kidney-treatment effect and investigate candidate arthritis treatments. The
analyst reads clinical records, runs an analysis, commissions a study and responds
to its findings. This demonstration uses a fixed script through the same tools
that an agent would use; no model API key is needed.

Use Python 3.11 or 3.12, uv and Docker. From a clone of the public harness repository,
install the locked environment and build the supplied executor:

```bash
uv sync --locked
docker build --tag trialagentbench/executor:1.0.1 executor
uv run trialagentbench demo run --output teaching-run
```

Run these commands from the public GitHub repository root. In the extracted toolkit, use its
wheel-install instructions, `execution` in place of `executor`, and the installed
`trialagentbench` command without `uv run`. The initial build needs dependency network
access; the clinical run itself is offline, with no provider credentials, provider extras or
benchmark download. Use a new output directory for a fresh run.

Open `teaching-run/index.html`. It displays estimates and 95% intervals beside known
references, numerical errors, loss, development decisions and regret, then study expenditure, programme time and
net development value. Observed patient rows and the submitted code are inspectable
on the page; complete CSV tables and native grades remain linked. Reproduced output
can still be scientifically poor.

Recover an interrupted run with:

```bash
uv run trialagentbench demo run --output teaching-run --resume
```

TrialDev resumes its saved acquisition
checkpoint. An unfinished TrialEval task restarts from its initial evidence after
retaining its entire earlier attempt under `failed_attempts/`, including code,
conversation, events and provider usage. The report links those retained attempts;
include their charges in total resource accounting. A completed run resumes without
more provider turns or acquisitions.

The TrialEval example rebuilds one released public C1 task of TE-S13-A1 for a single kidney
trial. Its bundled analysis table contains 128 participants and 640 repeated observations. The
TrialDev example uses the world of the first rheumatoid-arthritis task in the release's ablation
selection. Its observational cohort contains 2,400 participants and five treatment candidates
plus the comparator. The demo rebuild uses a 2,048-participant study pool, 16 observational reference
draws and quadrature order 8 to keep the archive small; study readouts and reference values therefore
can differ from the released world's. The initial participant records match the release's bytes.
`trialdev-specification.json` records this provenance and the bundled archive counts; it is not a
generation specification. Both examples publish their reference answers.
During the run, the controller receives only participant
messages and tool outputs. The harness alone loads the patient pool and numerical
references, preserving the distinction between evidence an analyst can inspect and
answers used for evaluation.

TrialEval analyses one kidney-disease trial with the plan's Gaussian marginal GEE, a linear
treatment-by-time interaction, exchangeable working correlation and participant-clustered sandwich
uncertainty at Day365. TrialDev analyses the programme's
2,400-person observational rheumatoid-arthritis cohort, validates and starts a 60-person
early-safety study, opening sites in the public activation order until they list 480
candidates and forecast 60 screened in time for its readout, acquires its Day56 observations, reports new
contrasts and stops. Its available patient pool has 2,048 people. The analyses are
deliberately elementary: TrialDev uses observed mean differences with Welch intervals and
adjusted risk differences with Agresti–Caffo intervals; neither analysis corrects missingness, and TrialDev does not adjust
confounding or standardise to the programme population. The native grader exposes
the resulting numerical errors and interval loss.
The run demonstrates the research interaction, without measuring AI performance.

`observed_analysis.py` is the actual saved, sandboxed and replayed analysis;
`choose_study.py` selects the study from public design rules. The scripted
`LLMProvider` in `trialagentbench.tools.demo` derives its next action from persisted
messages, uses no network client and reports zero provider tokens/cost. Real local
CPU, memory and process measurements remain recorded by the executor.

To integrate another agent, copy and run the complete external adapter:

```bash
uv run python examples/clinical/run_external_provider.py --output external-teaching-run
```

An installed wheel retains this directory under
`trialagentbench/resources/examples/clinical/`. `ExampleProvider.generate_turn`
shows the exact protocol. Replace its delegated fixed policy with your client;
return `LLMResponse`/`ToolCall` objects and report actual usage and routing metadata.
The `run_demo(provider=...)` call uses the harness's normal tools and grader.
Use the campaign commands in [RUNNING.md](../../docs/RUNNING.md) for benchmark task panels.
The bundled TrialEval reference is recomputed for this one trial by the benchmark's oracle, so it
agrees with the released value within Monte Carlo error. The public harness includes the prepared
archives; rebuilding them requires the private construction pipeline. Evaluator answers remain
in separate archives and are never passed to the scripted controller.
