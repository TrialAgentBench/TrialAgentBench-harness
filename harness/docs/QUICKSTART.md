# Quickstart: TrialAgentBench Harness

## 1. Get the dataset

Download the participant, evaluator, and verification archives from the
[TrialAgentBench dataset release](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/).
Saved run outputs are produced by running the harness.

The participant archive contains the task files shown to the agent. The
evaluator archive contains the analyses and decisions used for scoring. The
verification archive contains the independent reconstruction records.

The harness accepts:
- `--participant-dir <trialeval-participant-release-dir>` for live TrialEvalBench execution.
- `--suite-dir <TrialEvalBench_evaluator.zip>` for offline TrialEvalBench
  grading when the correspondingly named participant ZIP is beside it. An
  already extracted root containing `public/` and `grader/` is also accepted.
- `--bundle <trialdev-evaluator-dir>` for TrialDevBench. Extract the evaluator
  archive for the selected stream. The runner stages only participant-role
  files and currently available evidence into the model workspace.

## 2. Install

```bash
git clone https://github.com/TrialAgentBench/TrialAgentBench-harness.git
cd TrialAgentBench-harness
uv sync
# API keys are needed only for live model runs, not offline grading.
```

The harness wheel includes TrialDev execution, fixed-evidence replay, and
grading contracts; no construction package or local package checkout is
required.

Live execution additionally needs the locked `executor/` Docker build context.
It is included in the public source repository and source distribution, not
installed by the wheel. If you installed the wheel, obtain the matching source
archive or checkout before building the executor image.

From the repository root, build the exact image expected by the harness:

```bash
docker build -t trialagentbench/executor:0.1.0 harness/executor
```

Run the isolated-execution tests with:

```bash
make -C harness test-executor
```

Verify the install:

```bash
uv run python -c "from trialagentbench_harness.trialdev.grading.sequential import grade_trajectory_v1; print('ok')"
```

Core installation supports all run, grading, replay, and trace commands.
TrialEval participant-diagnostic generation additionally requires the analysis
extra; without it the CLI exits with an explicit installation instruction:

```bash
uv sync --extra analysis
```

For TrialDev execution, extract the evaluator archive for one of the two
separately reported streams. Pass that directory to the runner; the runner
stages only the participant role and only the evidence available at the current
checkpoint:

```bash
mkdir <trialdev-release-dir>
unzip <trialdev-stream-evaluator.zip> -d <trialdev-release-dir>

uv run trialagentbench run trialdev \
  --bundle <trialdev-release-dir> \
  --programs <programme-id-or-scenario:objective> \
  --provider openrouter \
  --model <exact-model-id> \
  --openrouter-provider <exact-upstream-provider>
```

The release contains fixed, checksummed phase evidence. A submitted design is
evaluated against the released design policy and frontier; it does not trigger
outcome generation. Evaluator and verification files remain outside the
model-facing workspace.

The default execution budgets are 90 turns per TrialEval item and 45 turns per
TrialDev semantic step. They are resource ceilings, not stopping guidance:
successful submission terminates the step immediately. Override them only as a
declared experimental condition; the exact values are stored in run provenance
and must be held fixed within a comparison.

The standard TrialDev interaction uses `--tool-choice auto`,
`--procedure-assistance output_contract_only`, and at most three corrected
portfolio submissions per checkpoint. `--tool-choice required`, checklist or
ordered assistance, and alternative turn or correction limits are explicit
experimental conditions rather than default guidance. Trial-materialisation
retries (`--max-phase-retries`) and corrected portfolio submissions
(`--max-submission-attempts`) are independent controls.

For a bounded live run, `--reported-cost-stop-usd <amount>` prevents new
provider requests after the cumulative provider-reported cost reaches the
threshold. Because response cost is observed after completion, concurrent
in-flight requests may overshoot it; `run_stop.json` reports the observed cost,
overshoot, and exact completed, interrupted, and unstarted schedule members.
An interrupted run preserves the same custody. The option is a run-control
threshold, not a hard pre-spend reservation.

For repeated or multi-model evaluation, place the same settings in a strict
machine-readable request and run it directly:

```json
{
  "schema_id": "trialagentbench.trialdev_execution_request/v1",
  "bundle": "./trialdev-release",
  "model": "provider/model-id",
  "provider": "openrouter",
  "openrouter_provider": "ExactUpstreamProvider",
  "condition_id": "model-primary",
  "request_replicate_id": "request-1",
  "programs": ["programme-id"],
  "master_seed": 45560,
  "max_turns_per_step": 45,
  "max_submission_attempts": 3,
  "procedure_assistance": "output_contract_only",
  "tool_choice": "auto",
  "output_root": "./results"
}
```

```bash
uv run trialagentbench run trialdev --experiment-config experiment.json
```

Relative bundle, capability-record, output, and continuation paths are resolved
from the configuration file. Unknown fields and contradictory provider or
reasoning controls fail before provider construction. Changing models requires a
new configuration value, not a source edit.

To continue a deliberately stopped schedule, use the same request with
`append_run_dir` set to the existing run directory. The runner validates the
release, model, condition, interface, source, seed, and complete selected
programme denominator before it skips completed programmes and runs only the
remaining members. It rejects a partial programme or any identity drift.

## 3. Offline Grading

Use the installed CLI for a run produced under the canonical release contract:

```bash
uv run trialagentbench grade trialeval <canonical-trialeval-run-dir> \
  --suite-dir <TrialEvalBench_evaluator.zip> --out-dir <graded-trialeval-run-dir>
uv run trialagentbench grade trialdev <canonical-trialdev-run-dir> \
  --bundle <extracted-trialdev-evaluator-release> --out-dir <graded-trialdev-run-dir>

uv run trialagentbench analyse trialdev-results \
  --input <graded-arm-a>/trialdev_assessments.json <graded-arm-b>/trialdev_assessments.json \
  --output <analysis-dir>/summary.json \
  --reference-condition <condition-a> \
  --intervention-condition <condition-b> \
  --comparison-output <analysis-dir>/comparison.json

# Build public trace-analysis tables and figures from supplied trace-bearing runs.
uv run trialagentbench analyse trace \
  --out-dir <trace-bundle-dir> \
  --trialeval-root <graded-trialeval-runs> \
  --trialdev-root <graded-trialdev-runs>
```

Offline grading does not import provider adapters, require API keys, or make
network calls. Keep each evaluator ZIP beside the correspondingly named
participant ZIP; the grader validates and combines the pair in a temporary
isolated workspace.

Export the complete reproduction record after run, grade, and independent
verification:

```bash
uv run trialagentbench export results \
  --release-root <extracted-release> \
  --run-root <canonical-run-dir> \
  --grade-root <graded-run-dir> \
  --verification-root <verification-output> \
  --analysis-root <trace-bundle-dir> \
  --output-dir <new-result-bundle-directory>
```

The exporter writes a deterministic archive, a detached checksum, and a
release-bound receipt. It refuses to read results from, or write results into,
the immutable release.

## 4. Live provider-backed runs

Install the provider extra before a live run. TrialEval receives the extracted
participant release shown below. TrialDev receives one extracted evaluator
archive as shown in section 2; its evaluator files remain on the host and are
never mounted into the model workspace. `--task-id` accepts an exact opaque
allowlist for bounded TrialEval runs; omit it for a complete TrialEval
participant release.

```bash
uv sync --extra providers
uv run trialagentbench run trialeval \
  --participant-dir <trialeval-release>/public \
  --task-id TASK0001 TASK0002 \
  --provider openrouter \
  --model <exact-model-id> \
  --openrouter-provider <exact-upstream-provider> \
  --item-watchdog-seconds 3600 \
  --workers 1 \
  --output-dir <new-run-root>
```

Each run records the participant-release checksum, exact task denominator, and
the analysis specification carried by every selected task. `protocol_only`
evaluates analysis selection from the clinical question and evidence;
`locked_sap` evaluates execution of a frozen primary analysis. The runner does
not accept an override that could change this participant evidence after item
construction. An OpenRouter provider pin disables route
fallback and the run fails if OpenRouter reports a different or missing
upstream identity. Provider-backed execution is not required to inspect or
grade an existing canonical run.

`--provider` identifies the API surface as well as the gateway:

| Value | API surface | Routing |
|---|---|---|
| `openai` | Chat Completions | Direct OpenAI |
| `openai_responses` | Responses | Direct OpenAI |
| `openrouter` | Chat Completions | OpenRouter with the exact required `--openrouter-provider` |

Chat Completions remains an explicit supported condition; selecting Responses
does not silently reinterpret an existing run. Responses execution sets
`store=false` and replays the complete returned output-item state locally,
including reasoning-state items required for valid tool continuations. The
current OpenAI Responses endpoint does not accept the harness decoding seed, so
do not combine `--provider openai_responses` with `--decoding-seed`.

The model is connected only to TrialAgentBench function tools. `execute_code`
runs inside the release Docker image with the evidence mounted read-only,
networking disabled, and a bounded writable scratch directory. No provider
hosted code interpreter is used.

## 5. Grade saved submissions

Live execution intentionally produces no score artifacts. Grade the immutable
run with the paired evaluator release before analysis:

```bash
uv run trialagentbench grade trialdev results/trialdevbench/<model-id>/<run-id> \
  --bundle <extracted-trialdev-evaluator-release> --out-dir <graded-run-dir>
```

This grades every programme submission and writes the typed
`trialdev_assessments.json` and `trialdev_metrics.json` surfaces, the exact run
configuration and coverage, per-programme evidence, and `GRADE_MANIFEST.json`
under the new graded output. The command publishes `--out-dir` transactionally
and refuses to overwrite an existing path. Use `analyse trialdev-results` to
combine named conditions and produce a paired comparison.

## 6. Where things live

| path | what |
|---|---|
| `<graded-run-dir>/trialdev_assessments.json` | one typed record per scheduled programme and condition |
| `<graded-run-dir>/trialdev_metrics.json` | stream-specific counts, denominators, component rates, and resources |
| `<graded-run-dir>/GRADE_MANIFEST.json` | checksums for the immutable input and graded output |
| `<analysis-dir>/comparison.json` | paired named-condition effects with scenario-cluster uncertainty |
| `results/trialdevbench/<model-id>/<run-id>/run_coverage.json` | immutable live-run denominator |
| `results/**/provider_telemetry_summary.json` | response counts, latency, routing, token use, and provider-reported cost |
| `results/trialdevbench/<model-id>/<run-id>/programs/<program_id>/` | per-program artefacts (conversation, grades, every submission) |

For the submission, result, checksum, and path contracts see
[CONTRACTS.md](CONTRACTS.md).
