# Contracts

All machine interfaces are strict, versioned Pydantic contracts. Unknown
fields, invalid enumerations, non-finite numbers, inconsistent checksums, and
unsafe paths fail validation.

## Release contracts

The core `RELEASE_MANIFEST.json` binds:

- release and source identities;
- exactly six participant/evaluator/verification archives;
- the 550-row simulation-properties catalogue and schema;
- the prospective 500-row TrialEval and 50-row single-asset TrialDev generation-unit inventories;
- the canonical result schema;
- the documentation index; and
- the prespecified crossed-cluster analysis contracts.

`metadata/simulation_properties.jsonl` contains one record for each of 500
TrialEval items and 50 single-asset TrialDev trajectories. Each single-asset
trajectory is evaluated under four disclosed programme objectives, yielding
200 programme views. Its stable keys distinguish:

- `analysis_unit_id`: released item or trajectory;
- `generation_unit_id`: generated trial or programme;
- `independence_unit_id`: inferential clustering unit;
- `matched_set_id`: dependent views used for within-set contrasts;
- `scoring_unit_ids`: exact routes or lanes; and
- `canonical_result_join_key`: many-to-one join from result rows to simulation
properties.

The catalogue describes construction and analysis properties. It contains no
scoring rules and is not participant-visible.

`metadata/trialeval_experiment_assignments.jsonl` and
`metadata/trialdev_experiment_assignments.jsonl` schedule every core generation
unit once before model execution. Each row has an immutable `assignment_id`,
the same `canonical_result_join_key` as its catalogue record, and status
`scheduled`. The TrialDev runner crosses each scheduled single-asset trajectory
with the four objective policies declared by its suite manifest.

The bounded-portfolio stream has a separate, checksummed scientific inventory:
12 evidence worlds crossed with four objective policies and two resource
budgets, or 96 programme-policy views. Its evaluator manifest binds 108 unique
asset-phase episodes, stores each episode once, and records the exact source
inventory included in the implementation hash. Objective and budget views of
the same world are matched views, not independent generated programmes.

## TrialEval submission

The authoritative public schema is `agent_output_schema.json`. A submission
identifies the task and declares one primary analysis containing:

- estimand: population, treatment, comparator, endpoint, intercurrent-event
  strategies, and horizon;
- estimator: one exact `analysis_method_id` selected from the public method
  dictionary, with optional audit-only implementation and qualification notes;
- result kind and result;
- favorable direction;
- evidence references; and
- optional audit-only free-text limitations.

A bounded-deviation primary result uses `result_kind: sensitivity_set` and a
typed vector. Its component names bind each prespecified sensitivity parameter
to one lower and one upper endpoint, for example `delta_0.05_lower` and
`delta_0.05_upper`. The selected method record fixes the complete, increasing
parameter grid. This makes the statistical departure model machine-checkable
without exposing an item-specific numerical answer.

C5 submissions additionally carry one `data_integrity_record` that declares
the applied repair and its evidence. When `data_integrity_policy.json` is
present, the harness offers `validate_data_integrity`. The agent must execute
the declared repair and write its repaired Parquet input under `scratch/`.
The public operation fails on ambiguous same-key records or an inexact repair
and returns the canonical record required by the submission schema. It has no
access to evaluator references, expected counts, clean-parent checksums, or
scores. The official benchmark endpoint is the structured submission.
Narrative-to-structured normalization is a separate,
prespecified interface experiment: its normalized record retains source-span
provenance, and its measurement error is qualified separately before any
interface contrast is interpreted.

`method_dictionary.json` in the TrialEval participant archive defines every
available method ID and its intrinsic family, result shape, effect scale,
design modifiers, uncertainty method, sensitivity grid, and possible public
diagnostics. The harness mounts this dictionary in every task workspace. It
contains no item-specific eligibility, required diagnostic set, estimand,
reference value, tolerance, accepted answer, or score.
Before a run, `trialagentbench verify submission --suite
{trialeval,trialdev} --submission FILE [--participant ZIP]` validates the
public schema and, when a participant ZIP is supplied, the task or scenario
identity. It never opens evaluator artifacts.

## TrialDev submissions

The single-asset stream uses phase-specific contracts for:

- observational analysis and asset nomination;
- randomized-phase design request;
- randomized-phase analysis;
- checkpoint decision;
- terminal recommendation; and
- safety evidence.

Randomized requests contain exactly one investigational regimen; control is
defined by the public phase policy. A request that violates the phase policy is
rejected before execution. Released runs replay checksummed fixed phase
evidence; the public harness does not regenerate outcome data.

The bounded-portfolio stream uses one typed checkpoint record containing:

- the current programme-state checksum;
- a reproducible statistical conclusion for the available evidence;
- one selected action from the current feasible action policy;
- the exact evidence references used; and
- one precommitted design-cell identifier for each scheduled study.

The grader independently recomputes identification, estimates, intervals,
safety classification, supported actions, design feasibility, and resource
feasibility from released evidence. An identified but imprecise comparison may
support several actions. A nonidentified observational comparison supports a
qualified withholding action and does not admit invented candidate effects.
Identification is assessed under the method's declared assumptions and public
treatment-assignment provenance; it is not a claim that exchangeability was
empirically proven. The public method record fixes the bootstrap seed, random
number generator, standard-error convention, and interval construction used
for numeric conformance.

Every portfolio submission attempt is retained outside the participant
workspace with its exact payload and one typed transport outcome: contract
rejection or acceptance. A contract-valid legal decision advances the
programme. The independent grade records whether its analysis, uncertainty,
action, and next-study design are scientifically supported; a weak grade is a
benchmark result, not a request to reproduce an evaluator-held answer.

## Result rows

The canonical downstream result schema is
`metadata/canonical_result.schema.json`. Each row binds release, suite,
analysis unit, scoring unit, model/provider snapshot, run and assignment
identity, decoding replicate and seed, assistance/interface/budget conditions,
completion state, optional route or action, result, and component/final grade.

A non-completed row must have one failure code. Interval endpoints must occur
together and be ordered. Joining results to simulation properties uses
`canonical_result_join_key`; every result row must match exactly one catalogue
row.

## Result export

`metadata/export_results_manifest.json`, bound by the core release manifest,
declares the canonical result schema, the 550-generation-unit core inventory,
and the required run, grade, and independent-verification artifact kinds.
TrialDev result rows additionally retain stream, programme-view, objective,
budget, checkpoint, and trajectory identities so the 200 single-asset and 96
portfolio programme views are reported separately without treating matched
views as independent. The executable projection is:

```bash
trialagentbench export results \
  --release-root <extracted-release> \
  --run-root <run-output> \
  --grade-root <grade-output> \
  --verification-root <verification-output> \
  --output-dir <new-result-bundle-directory>
```

Each input is a caller-owned directory outside the immutable release. The
exporter rejects links, special files, credential files, transient cache
directories, overlapping inputs, and output reuse. It writes one deterministic
ZIP containing `manifest.json`, `SHA256SUMS`, and the selected artifacts, plus
a detached ZIP checksum and `result_export_receipt.json`. The receipt binds the
release manifest, export contract, source labels, member paths, sizes, and
checksums without recording local absolute paths.

## Checksums and paths

JSON checksums use canonical UTF-8 serialization with sorted keys, no NaN or
infinity, and the checksum field excluded from its own digest. JSONL record
order is part of the artifact identity where the manifest declares it.

Release-relative paths must be nonempty, relative POSIX paths without `..`.
ZIP readers reject absolute paths, traversal, links, duplicate normalized
members, case collisions, and unsupported compression.

## Compatibility

Contract versions are explicit. The release does not silently translate
superseded identities or accept mixed-version records. A score-affecting schema
or route change creates a new release identity and requires regeneration of
dependent artifacts.

## Observable trace bundle

Trace analysis operates only on completed user-owned runs and does not infer
private reasoning. The analyzer discovers the observed model set; model names,
counts, and ordering are not fixed. A trace bundle contains these seven
schema-backed tables:

- `action_events.csv`;
- `evidence_use.csv`;
- `failure_cascades.csv`;
- `semantic_features.csv`;
- `trialdev_phase_outcomes.csv`;
- `trialdev_program_cascades.csv`; and
- `unit_features.csv`.

`trace_bundle_manifest.json` binds their checksums and schemas. Trace features
are descriptive process measurements and do not alter endpoint grades.
