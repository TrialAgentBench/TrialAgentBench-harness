# Controlled comparisons

The experiment layer measures changes in agent process while holding the
scientific task and grading contract fixed. Experiment assignments never alter
an item's eligible routes, comparison tolerances, or evaluation target.

## TrialEval primary experiment

The primary experiment estimates the effect of procedural assistance on
analysis conformance. It crosses:

- three assistance conditions;
- structured and narrative response interfaces; and
- two prespecified decoding replicates.

One context view is selected for each of the 100 independent base trials by a
frozen regime-cell-stratified allocation. Each mechanism cell retains four
different context views, and each context is retained equally often overall. All conditions are
crossed within the selected base-trial view, so comparisons use paired
base-trial differences rather than treating assignments as independent.

### Assistance conditions

- **P0, output contract only:** required output fields and machine schema.
- **P1, unordered checklist:** the same analysis operations as P2 without a
  required order.
- **P2, ordered procedure:** a fixed order for defining the question,
  inspecting evidence, assessing assumptions, executing the analysis, checking
  uncertainty, and verifying the report.

P1 and P2 contain the same task-general operations. Their contrast isolates
ordering; it is not a contrast between fewer and more statistical hints.

### Analysis-specification conditions

`protocol_only` supplies the scientific question and study protocol but leaves
selection of a supported analysis route to the agent.

`locked_sap` supplies the complete prespecified analysis plan: estimand, effect
scale, analysis method, uncertainty procedure, diagnostics, and sensitivity
obligations. It does not prescribe an ordered work process.

The locked-SAP versus protocol-only contrast therefore measures the value of
analysis specification, while P2 versus P1 measures the value of ordering.

### Response interface

The structured interface writes directly to the submission schema. The
narrative interface produces a report that is transcribed into that schema.
The primary response-interface contrast is
`structured-minus-narrative`.

Narrative transcription is a measurement step, not a statistical analysis
route. Before use, the transcriber is evaluated on a frozen, outcome-blind,
stratified sample against masked human reference transcriptions. The error event
is any disagreement in a score-relevant field. The design, sample size,
acceptance count, seed, and secondary interval procedure are stored in the
frozen experiment design.

Every scored endpoint records whether normalization was complete, abstained,
or did not apply; any abstention reason; any required semantic deliverables
that were absent; and the deterministic primary failure code. A normalizer
abstention or a correctly transcribed participant omission remains in the
scheduled denominator. Invalid evaluator contracts and checksum or schema
drift still terminate grading.

### Primary estimand

The primary estimand is the paired mean difference in
`primary_analysis_conforms` for P2 minus P0 among all scheduled factorial
assignments. Noncompletion is failure. The unit of independence is the base
trial; decoding repeats and crossed conditions are repeated measurements.

Supporting endpoints include:

- usable primary analysis;
- fixed-question and analysis-method match;
- required obligations met;
- numeric result agreement where applicable;
- uncertainty agreement;
- planning consequence where applicable; and
- typed failure class.

Planning endpoints use the
`planning_consequence_evaluable_assignments` population because planning is not
defined for every trial question. This restriction is prospective and does not
remove failed assignments for otherwise planning-eligible questions.

### Statistical analysis

Publication contrasts use the frozen analysis configuration:

- paired differences at base-trial level;
- equal weight per independent base trial;
- all scheduled pairs retained;
- 10,000 crossed-cluster bootstrap resamples;
- 95% intervals; and
- the declared bootstrap seed.

The single primary contrast does not require multiplicity adjustment.
Supporting endpoints and subgroup interactions are reported as such.

### Provider conditions

Live TrialEval runs bind the model route, requested reasoning effort,
procedure assistance, turn ceiling, condition identifier, and provider-request
replicate in one immutable condition record. A reasoning-effort request
requires a checksum-bound capability record for the exact transport, model,
and upstream provider. The run fails before credential loading or output
creation when these identities disagree.

```bash
trialagentbench run trialeval \
  --participant-dir PARTICIPANT_RELEASE \
  --provider openrouter \
  --openrouter-provider OpenAI \
  --model provider/model-id \
  --condition-id reasoning-high \
  --request-replicate-id request-1 \
  --reasoning-effort high \
  --reasoning-capability-snapshot provider-reasoning-capability.json \
  --turns 45 \
  --omit-temperature
```

For a matched reasoning-effort comparison, hold the participant release,
task set, prompt, tools, turn ceiling, output limit, provider route, and
request-replicate schedule fixed. Vary only the requested effort and condition
identifier. Tasks are the analysis units for a finite benchmark panel;
provider-request replicates are repeated measurements of the same task.

## Context contrasts

The five context views answer the same statistical question. Prespecified
paired contrasts isolate evidence availability:

- C1 versus C2: locked analysis specification with analysis-ready data versus
  protocol-only specification with analysis-ready data;
- C3 versus C4: the same specification contrast with less processed data;
- C3 versus C1: data preparation under a locked analysis specification;
- C4 versus C2: data preparation under protocol-only specification; and
- C5 versus C4: the effect of one exactly repairable integrity condition.

Context comparisons must preserve the base-trial match and may not treat the
five views as independent datasets.

## Design and assumption subgrouping

Subgroup summaries use the canonical evaluation-series, design-subtype, and
assumption-tier fields. The D label alone is not an analysis method. In
particular, D1 context changes do not identify an ordinal PH-dose effect; a
proportional-hazards claim requires the endpoint, contrast, and assumption state
that actually support it.

A4 items are summarized by their declared broken default and admitted response:
alternative estimator, sensitivity set, identification bound, qualified
non-estimability, or abstention. These categories must not be collapsed into a
generic refusal rate.

## TrialDev experiments

TrialDev experiment rows retain the whole programme trajectory or portfolio
evidence world as the independence unit. The two streams are analysed and
reported separately.

Provider reasoning effort can be varied as an experimental condition when the
exact model route exposes that control. The run command requires a
checksum-bound capability record and rejects a requested effort if the record's
transport, model, upstream provider, or supported values differ from the run.
The requested effort is the intervention; private reasoning content is neither
collected nor treated as evidence that the intervention was applied. Model,
condition, request replicate, and task-materialization seed remain separate
fields from run through grade, analysis, comparison, and export.

TrialDev comparisons pair named conditions, not model-name aliases. This permits
comparisons such as low versus high reasoning effort for the same model while
also retaining the exact model and route in provenance. Request replicates are
repeated observations against one task instance; they are not independent
evidence worlds.

Scaled runs can use the strict
`trialagentbench.trialdev_execution_request/v1` JSON contract described in the
quickstart. The request contains model, route, selected programmes, condition,
request replicate, task seed, interface, and run-control settings. Relative paths
are configuration-relative, and unsupported fields or route combinations fail
before any provider request. The same contract applies to every model; no model
identifier selects a source-code branch.

```bash
trialagentbench run trialdev \
  --bundle RELEASE_ROOT \
  --provider openrouter \
  --openrouter-provider OpenAI \
  --model provider/model-id \
  --condition-id reasoning-high-p0 \
  --request-replicate-id request-1 \
  --reasoning-effort high \
  --reasoning-capability-snapshot provider-reasoning-capability.json \
  --master-seed 45560 \
  --omit-temperature
```

Omit `--reasoning-effort` and its capability record for an uncontrolled
reasoning condition. Provider request replication does not imply seeded model
determinism, so `--decoding-seed` is omitted unless the exact endpoint contract
supports and the experiment declares it.

Turn ceilings and procedure assistance are also explicit condition factors.
Use `--max-turns-per-step` and `--procedure-assistance` prospectively and keep
both fixed for a reasoning-effort contrast. The assistance levels expose the
same public evidence and submission contracts: `output_contract_only` states
the required output, `unordered_checklist` lists the analysis operations, and
`ordered_sop` orders those same operations. They do not provide fitted values,
accepted actions, preferred estimators, grading tolerances, or future evidence.
Select a turn/scaffold setting from a bounded calibration using validity and
completion first and resource use second; report the selected setting as local
to the evaluated tasks and route rather than universally optimal.

Tool policy is a separate interface factor. `--tool-choice auto` is the clean
benchmark condition because it leaves analysis and tool-use strategy to the
model. `--tool-choice required` is an ablation and must not be relabelled as the
default task. Hold `--max-submission-attempts` fixed when estimating turn or
prompt effects; it controls corrected submissions, while
`--max-phase-retries` controls rejected trial-materialisation requests.

For the 200 single-asset programme views, the main contrasts are:

- procedural assistance;
- observational-analysis specification;
- checkpoint replay.

Assignments are crossed within each of the 50 environment–trajectory scenarios
and four programme objectives.
Endpoint rows retain all seven scoring lanes and the checkpoint at which each
decision was made. A later result cannot be used to rescore an earlier
decision.

For the bounded-portfolio stream, 12 independent evidence worlds are crossed
with four objective policies and two resource budgets. The 96 resulting views
are paired policy questions. Analyses cluster uncertainty by evidence world and
retain objective and budget matching. Primary outcomes include complete-
programme success, cumulative success through each checkpoint, supported-set
size, allocation or withholding, reserve promotion, permanent safety
exclusion, resource use, and typed noncompletion. These results are never
combined with the single-asset stream into one scalar.

TrialDev v1 uses typed, structured decisions at every checkpoint. It does not
schedule a narrative response interface or post-hoc narrative normalization.
Each checkpoint is validated and graded independently from the evidence that
was available at that time. This preserves the temporal decision task and
prevents later evidence from repairing an earlier decision. A future narrative
TrialDev experiment would require its own prospective checkpoint-level
normalization qualification before comparison with the structured interface.

The observational specification experiment concerns adjustment,
identification, and uncertainty in the nonrandomized screen. It does not change
the later randomized comparison. Single-asset phases identify the nominated
regimen and control explicitly; portfolio phases identify the reached active
asset or reached lead–reserve pair and their concurrent controls.

## Schedule integrity

Every schedule binds:

- release and participant-archive identity;
- experiment-design and analysis-configuration checksums;
- task or scenario selection;
- assistance and interface condition;
- model and provider snapshot;
- condition and request-replicate identifiers;
- task-materialization, optional provider-decoding, and analysis seeds as
  distinct quantities;
- randomization seed;
- resource and budget profile; and
- expected denominator.

Schedules are immutable after model output exists. Missing or failed assignments
remain rows with typed completion states.

## Interpretation

Experiments estimate changes in benchmark process reliability under the
declared conditions. They do not establish that one prompt is universally
better, that a reference analysis is uniquely correct, or that a synthetic
trial reproduces every feature of a real development programme. Single-seed
results demonstrate execution and deterministic replay; repeated-world
evaluation is required for operating-characteristic claims.

## Semantic report assessment

TrialEval narrative reports may also be assessed by a reference-blind semantic
assessor. The assessor receives the task material available to the participant,
the method dictionary, the output contract, and line-numbered report text. It
records a noncompensatory result over question, method, evidence, integrity,
result structure, and result support. Every substantive finding cites the
supporting report lines.

The analysis reports agreement, false acceptance, false rejection,
invalid-response frequency, latency, and cost against a masked reference. These
metrics characterize the reliability of semantic report assessment and
narrative normalization.

Export the frozen, reference-blind packet set once from a completed narrative
run, then run either measurement process against the same packets:

```bash
trialagentbench export trialeval-narrative-packets \
  RUN_DIR PARTICIPANT_RELEASE_DIR PACKET_DIR

trialagentbench run trialeval-narrative-normalizer \
  PACKET_DIR NORMALIZATION_DIR \
  --provider openrouter --openrouter-provider PROVIDER --model MODEL

trialagentbench run trialeval-direct-assessment \
  PACKET_DIR ASSESSMENT_DIR \
  --provider openrouter --openrouter-provider PROVIDER --model MODEL
```

Both batch commands bind the packet manifest and provider configuration,
retain one typed result per scheduled packet, checksum the complete
denominator, and require `--resume` to reuse an interrupted output directory.
Neither command receives evaluator references or changes deterministic grades.
