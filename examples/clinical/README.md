# Work through a clinical research episode

The fictional Ariadne Clinical Research Collaborative asks an analyst to estimate
a kidney-treatment effect and investigate candidate arthritis treatments. The
analyst reads clinical records, runs an analysis, commissions a study and responds
to its findings. This demonstration uses a fixed script through the same tools
that an agent would use; no model API key is needed.

Install the public harness and its two runtime dependencies, and build the supplied
Docker executor. No provider credentials, provider extras, construction package or
full benchmark corpus are needed to run:

```bash
docker build --tag trialagentbench/executor:1.1.1 executor
trialagentbench demo run --output teaching-run
```

The build command assumes the public GitHub checkout. In the extracted toolkit,
use `execution` in place of `executor`. The initial build needs dependency network
access; the clinical run itself is offline.

Open `teaching-run/index.html`. It displays estimates and 95% intervals beside known
references, numerical errors and loss, then study expenditure, programme time and
net development value. Observed patient rows and the submitted code are inspectable
on the page; complete CSV tables and native grades remain linked. Reproduced output
can still be scientifically poor.

Recover with the same command and `--resume`. TrialDev resumes its saved acquisition
checkpoint. An unfinished TrialEval task restarts from its initial evidence after
retaining its entire earlier attempt under `failed_attempts/`, including code,
conversation, events and provider usage. The report links those retained attempts;
include their charges in total resource accounting. A completed run resumes without
more provider turns or acquisitions.

The example records and reference answers are public and separate from benchmark
evaluation tasks. During the run, the controller receives only participant
messages and tool outputs. The harness alone loads the patient pool and numerical
references, preserving the distinction between evidence an analyst can inspect and
answers used for evaluation.

TrialEval analyses one kidney-disease trial with a linear treatment-by-time model
and participant-clustered uncertainty at Day365. TrialDev analyses a 300-person
observational rheumatoid-arthritis cohort, validates and starts a 60-person
early-safety study across 12 sites, acquires its Day56 observations, reports new
contrasts and stops. Its available patient pool has 512 people. The analyses are
deliberately elementary: TrialDev uses observed mean/risk differences with Welch
intervals; neither analysis corrects missingness, and TrialDev does not adjust
confounding or standardize to the programme population. The native grader exposes
the resulting numerical errors. The resulting estimates and intervals show what those choices cost in accuracy.
The run demonstrates the research interaction, without measuring AI performance.

`observed_analysis.py` is the actual saved, sandboxed and replayed analysis;
`choose_study.py` selects the study from public design rules. The scripted
`LLMProvider` in `trialagentbench.tools.demo` derives its next action from persisted
messages, uses no network client and reports zero provider tokens/cost. Real local
CPU, memory and process measurements remain recorded by the executor.

To integrate another agent, copy and run the complete external adapter:

```bash
python examples/clinical/run_external_provider.py --output external-teaching-run
```

An installed wheel retains this directory under
`trialagentbench/resources/examples/clinical/`. `ExampleProvider.generate_turn`
shows the exact protocol. Replace its delegated fixed policy with your client;
return `LLMResponse`/`ToolCall` objects and report actual usage and routing metadata.
The `run_demo(provider=...)` call uses the normal native agent and grading owners.
Use the campaign commands in the [quickstart](../../docs/QUICKSTART.md) for benchmark task panels.


The bundled TrialEval reference uses 200 complete-world draws. TrialDev uses
16 observational reference draws and quadrature order 8. Evaluator answers
remain in the separate evaluator archives; they are not part of the scripted
controller.
