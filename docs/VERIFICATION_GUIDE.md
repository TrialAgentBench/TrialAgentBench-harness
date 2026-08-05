# Verification evidence

TrialAgentBench uses synthetic trials to change a feature that determines which
analyses are supported while holding the treatment question and remaining
evidence fixed. This makes agent responsiveness measurable, but synthetic
construction alone does not establish that the benchmark is valid. An external
researcher must be able to inspect how task evidence produces a grade,
determine whether the generated records preserve the structures required for
analysis, and test whether the capability being graded affects later
development decisions.

All verification commands run without a model API key.

## Claims and evidence

The checks follow that argument from evidence to analysis and then to decision.

| Claim | Question | Primary evidence |
|---|---|---|
| Inspectable grading | Can an external researcher reproduce the analysis or decision used to grade a response from the released task files? | [`trialeval/`](../verification/src/trialagentbench_validation/trialeval/) and [`trialdev/`](../verification/src/trialagentbench_validation/trialdev/) independently repeat the accepted analyses and supported decisions. Their records connect the supplied data, analysis, estimate, uncertainty, and grading result. |
| Analysis-relevant trial data | Do the generated records preserve the outcome, follow-up, and within-participant relationships used by clinical-trial analyses? | The [source-trial anchoring chapter](../verification/validation_results/reports/source-trial-anchoring.md) compares PATENCY, TERECO, and HeadSOAR at the source-trial scale. The participant-linkage controls show why plausible columns alone are insufficient: keeping records intact reduced correlation error by 30% and adjusted-treatment bias by 58% compared with independently sampling columns. |
| Response to controlled changes | When a trial design or statistical assumption changes, does an independent analysis show the expected consequence? | The [trial-design and assumption](../verification/validation_results/reports/trial-design-and-assumption-response.md) and [mechanism and effect-recovery](../verification/validation_results/reports/mechanism-and-effect-recovery.md) chapters test each controlled change, with complete estimates in [`RESULTS.csv`](../verification/validation_results/RESULTS.csv). |
| Development consequences | Does using the estimated effect together with its uncertainty retain better development options than using the estimate alone or an arbitrary label? | The [TrialDev decision report](../verification/validation_results/trialdev_v1/REPORT.md#decision-consequences) compares the three strategies across 7,200 simulated programmes under the same budget limits. Their mean probabilities of successful programme completion are 0.712, 0.656, and 0.653. |

## Units and denominators

The release uses the following names consistently.

| Name | Definition | Count |
|---|---|---:|
| TrialEval base trial | One independently generated clinical trial | 100 |
| TrialEval context view | One of five matched evidence presentations of a base trial | 500 |
| TrialDev single-asset trajectory | One candidate's evidence and decision sequence | 50 |
| TrialDev single-asset programme view | One trajectory evaluated under one of four objectives | 200 |
| TrialDev portfolio evidence world | One lead-reserve evidence configuration | 12 |
| TrialDev portfolio view | One evidence world crossed with an objective policy and resource budget | 96 |
| TrialDev randomized episode | One released randomized-phase evidence episode | 108 |
| Decision simulation programme | One programme in the decision-consequence analysis | 7,200 |

"Programme" refers to a TrialDev decision sequence. TrialEval is described in
base trials and context views so that matched presentations are not counted as
independent trials.

## Evidence trace

1. Read [`BENCHMARK_AND_DATA.md`](../harness/docs/BENCHMARK_AND_DATA.md) for
   the TrialEval and TrialDev questions and the role of each file.
2. Read [`SCORING.md`](../harness/docs/SCORING.md) for how the benchmark accepts
   multiple correct analyses, handles data that cannot support one causal
   answer, and uses uncertainty to determine which actions are supported.
3. Locate the claim in the
   [statistical report](../verification/validation_results/REPORT.md), then
   follow its chapter to the exact source, method, figure, result row, and
   supporting data.
4. Use the reconstruction commands for benchmark answers, the packaged
   simulation evidence for data and mechanism validation, or the TrialDev
   decision audit for development consequences.

The corresponding machine-readable benchmark contracts are in
[`harness/trialagentbench_harness/contracts/`](../harness/trialagentbench_harness/contracts/).
The statistical methods and uncertainty calculations are defined in
[`METHODS.md`](../verification/validation_results/METHODS.md), and
[`SOURCES.md`](../verification/validation_results/SOURCES.md) identifies the
external datasets.

## Package integrity

```bash
uv sync --all-packages --all-extras
uv run --package trialagentbench-validation python -c \
  "from pathlib import Path; from trialagentbench_validation.contracts.simulation_validation_bundle import SimulationValidationBundleV1, verify_simulation_validation_bundle; root = Path('verification/validation_results'); bundle = SimulationValidationBundleV1.model_validate_json((root / 'validation_bundle.json').read_text()); verify_simulation_validation_bundle(root, bundle); print(bundle.checksum)"
```

The command compares every packaged report, table, figure, and source record
with the validation inventory.

## Verification commands

| Claim | Command |
|---|---|
| Inspectable grading | `trialagentbench-validate trialeval-replay --help` and `trialagentbench-validate trialdev-replay --help` |
| Analysis-relevant trial data | Packaged results and inputs in `verification/validation_results/` |
| Response to controlled changes | Packaged results and inputs in `verification/validation_results/data/` |
| Development consequences | `trialagentbench-validate trialdev-policy-value-audit --help` |

Data-specific commands require the matching participant, evaluator, and
verification archives. They stop on missing files, mixed identities, checksum
disagreement, or numerical disagreement beyond the prespecified tolerance.

## Software checks

```bash
make check
make test
make build
make smoke
```

These checks cover accepted and rejected analysis routes, nonidentified cases,
deterministic phase replay, portfolio resource constraints, independent
reconstruction, source-scale comparisons, and package boundaries.
