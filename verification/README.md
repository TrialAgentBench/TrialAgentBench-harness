# TrialAgentBench verification

Synthetic trial data support an evaluation only when they preserve the
structure required by the analysis and when the graded conclusion can be
recovered from the evidence shown to the agent. `trialagentbench-validation`
tests both requirements.

The package reads the trial records and independently repeats the analyses and
development decisions used for grading. Its output shows how the supplied data,
chosen analysis, estimate, uncertainty, and supported action relate to one
another. Separate simulation experiments test whether the synthetic trials
preserve source-trial outcomes and participant relationships, and whether a
controlled change in the evidence produces the expected analytical
consequence.

The public harness is included in this repository under
[`harness/`](../harness/).
Benchmark data and the validation distribution are available from the
[TrialAgentBench dataset](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/).

## Claims and evidence

Independent reconstruction, simulation validity, and development consequences
answer different questions about the benchmark. The
[verification guide](../docs/VERIFICATION_GUIDE.md) links each claim to its
released units, code, and numerical results.

The [statistical report](validation_results/REPORT.md) tests whether synthetic
records preserve the outcome, follow-up, and participant relationships required
by clinical-trial analyses, and whether controlled changes have the expected
statistical consequences. The
[methods](validation_results/METHODS.md) define their estimands and uncertainty
calculations, [RESULTS.csv](validation_results/RESULTS.csv) is the complete
numerical index, and [sources](validation_results/SOURCES.md) identify the
external datasets. The
[TrialDev decision report](validation_results/trialdev_v1/REPORT.md) compares
the consequences of evidence-qualified, point-estimate, and label-based
decisions. The reconstruction modules and commands below recover the benchmark
answers from the data shown to the agent.

## Installation

From the repository root:

```bash
uv sync --all-packages --all-extras
uv run --package trialagentbench-validation trialagentbench-validate --help
```

The verification package can also be built and installed independently with
standard Python packaging tools.

Independent statistical replay uses only the base installation. Commands that
deliberately exercise the released grader as a black box also require the
public harness. From the repository root, install both local packages with:

```bash
python -m pip install ./harness "./verification[grader-controls]"
```

Those grader-control results test implementation behaviour. They are reported
separately from the statistical reconstructions, which never call the grader.

Regenerate the report figures directly from the packaged CSV tables:

```bash
python -m trialagentbench_validation.validation_figures.report \
  --validation-root verification/validation_results \
  --output-dir validation_figures
```

The cluster dose-response tables are independently reproducible:

```bash
python -m trialagentbench_validation.characterisation.cluster_response \
  --output-dir cluster_response
```

The stepped-wedge calendar-trend response is generated in the same form:

```bash
python -m trialagentbench_validation.characterisation.stepped_wedge_response \
  --output-dir stepped_wedge_response
```

## Packaged evidence

The wheel contains the report, methods, numerical tables, SVG and PDF figures,
and the CSV data used to draw them. Verify every file against the packaged
inventory:

```python
from trialagentbench_validation.external.release.bundle import (
    installed_validation_root,
    verify_installed_validation_bundle,
)

bundle = verify_installed_validation_bundle()
print(installed_validation_root() / bundle.report.relative_path)
```

## Input identity

Each reconstruction record identifies its input archives and their checksums,
so files from different releases cannot be combined accidentally. The packaged
simulation report is bound to the validation package through
`validation_bundle.json` and its recorded dependency-lock checksum.

Archive reconstruction establishes the supported answers for the supplied
benchmark data. Simulation experiments establish the behaviour of the
generating mechanisms across repeated source-sized trials.

## Independent reconstruction

The participant archive contains the data shown in each task, the evaluator
archive contains the analyses used for scoring, and the verification archive
contains the independent reconstruction records. The recoverability command
recalculates each analysis from the participant data and then checks agreement
with the scoring result.

```bash
trialagentbench-validate recoverability \
  --participant-release TrialEvalBench_participant.zip \
  --evaluator-release TrialEvalBench_evaluator.zip \
  --verification-release TrialEvalBench_verification.zip \
  --workers 4 \
  --output-dir recoverability
```

The command reports every analysis that can contribute to the score and stops
on incomplete inputs or numerical disagreement beyond the released tolerance.

For a complete matched set of release archives, one command runs the TrialEval
base-trial census,
independent route replay, TrialDev recovery, grader-behaviour checks, context
and leakage checks, and result-figure rendering:

```bash
trialagentbench-validate candidate-release \
  --release-root <release-root> \
  --output-dir release_validation \
  --verifier-lock poetry.lock
```

The output is checksum-bound to the release identity and includes tidy CSV
data, SVG/PDF/PNG figures, methods, results, sources, and an exact membership
manifest. These results describe the released datasets; repeated-world
properties such as bias, coverage, error control, and power are reported in
the simulation-validation results.

Focused commands expose the same checks:

| Command | Scientific purpose |
|---|---|
| `trialeval-replay` | Recompute TrialEval point estimates and uncertainty from participant-facing tables. |
| `trialeval-sentinels` | Check that every selected task exposes the data and analysis route required for replay. |
| `trialeval-c5-integrity` | Repair each declared transport duplicate and establish equality with its matched C4 data. |
| `trialdev-replay` | Refit the TrialDev observational analyses and reproduce candidate rankings and actions. |
| `trialdev-sentinels` | Reconstruct the prespecified high-risk TrialDev cases. |
| `trialdev-phase-replay` | Recompute randomized-phase efficacy, safety, allocation, and power results. |
| `trialdev-policy-value-audit` | Independently reconstruct the terminal success, resource use, reference-action coverage, and regret associated with each supported action. |
| `trialdev-scientific-package` | Build the self-contained TrialDev methods, results, and validated vector-figure package. |
| `trialdev-scientific-package-verify` | Recompute the scientific-package manifest and decision-consequence analysis from packaged inputs. |
| `grader-concordance` | Compare separately implemented grade records with the installed public harness. |
| `grader-behavior` | Exercise accepted, alternative, rejected, abstaining, malformed, and non-identified submissions. |
| `worked-example` | Export a fully reconstructed example with its public inputs and result. |

Run `trialagentbench-validate <command> --help` for the exact inputs.

## Simulation validity

The simulation checks answer three different questions that must remain
separate.

**Outcome replication** asks whether generated trials reproduce clinically
recognisable outcomes at the source trial's sample size. Survival analyses
compare complete event-free curves and risk sets, ordinal analyses compare all
outcome categories, and longitudinal analyses compare trajectories,
attendance, and within-participant relationships.

**Mechanism recovery** changes one known generating parameter and asks whether
a separately fitted analysis recovers the change. A unit-scale recovery slope
of one means that a one-unit change in the generating parameter produces a
one-unit change in the estimate. Null settings measure false-positive behaviour,
while larger sample sizes distinguish an incorrect mechanism from limited
information.

**Structure controls** preserve the individual values but deliberately attach
them to the wrong participants, visits, treatment arms, or event times. A
material change in the analysis after this intervention identifies the
contribution of participant linkage.

The main simulation commands are:

| Command | Process tested |
|---|---|
| `survival-validation` | Event times, censoring, survival curves, and treatment hazards. |
| `ordinal-validation` | Complete ordered outcomes, safety risks, and treatment odds ratios. |
| `clustered-ordinal-validation` | Variable cluster sizes, within-cluster dependence, and robust treatment analyses. |
| `longitudinal-validation` | Repeated trajectories, within-participant correlation, and treatment contrasts. |
| `multivariate-longitudinal-validation` | Several jointly generated outcomes measured over time. |
| `longitudinal-observation` | Outcome-dependent dropout and the resulting bias in treatment trajectories. |
| `hte-validation` | Treatment effects that vary with a baseline characteristic. |
| `competing-risk-validation` | Primary and competing events whose probabilities constrain one another. |
| `confounding-validation` | Exposure imbalance, limited overlap, and adjustment or weighting recovery. |
| `native-stress-recovery` | Time-varying hazards and repeated-event heterogeneity. |
| `rctbench-validation` | Source-sized generation across 10 public participant-level trials. |

### Mechanism response

The observation-process analysis asks a practical question: when dropout
depends on a participant's previous outcome, does the generated dataset contain
that dependence, and does an analysis that models attendance reduce the
resulting treatment-effect error? The verifier estimates the dropout response
directly, then compares ordinary available-case analysis with inverse-
probability weighting. Across the two public longitudinal sources, recovered
dropout slopes were 1.044 (95% interval 1.004 to 1.084) and 0.999 (0.985 to
1.014) per unit configured effect. Weighting materially reduced error in two
of the three informative settings; the remaining comparison was compatible
with no improvement.

The recurrent-event analysis asks whether some participants can have
persistently higher event rates than others, as observed in real safety data.
Fifteen ImmPort studies supplied the external reference. Their equal-study
heterogeneity estimate was 0.451 (interval 0.369 to 1.162). Across configured
heterogeneity levels, the fitted profile-likelihood analysis recovered 0.964
units (0.947 to 0.981) per configured unit, while a direct check of the generating
law recovered 1.013 (0.986 to 1.040). The direct moment identifies the
generating behaviour; the fitted result measures finite-sample recovery.

The competing-risk analysis asks whether increasing one event type has the
correct consequence for another. At source-trial sample sizes, the recovered
primary- and competing-cause coefficient slopes were 1.069 (0.993 to 1.145)
and 1.004 (0.916 to 1.092). Increasing the competing-event coefficient reduced
primary-event probability by 0.039 (0.029 to 0.048) and increased any-event
probability by 0.076 (0.067 to 0.085) per unit change. The signs and magnitudes
therefore agree with the mechanism rather than treating the two outcomes as
independent.

## External trial comparison

AACT and RCTBench provide external distributions for observable trial
properties, including age, body mass index, enrollment, follow-up, and event
frequency.

The command consumes three released JSON contracts. The source manifest
identifies immutable inputs, the construct map defines comparable variables and
units, and the design declares study partitions and uncertainty calculations.
The workflow records their checksums and does not infer them from local
filenames.

```bash
trialagentbench-validate external-validate \
  --source-manifest verification/external_validation/quantitative_source_manifest.json \
  --construct-map verification/external_validation/construct_map.json \
  --design verification/external_validation/validation_design.json \
  --aact 20260701_export_ctgov.zip \
  --rct-bench rct_bench-main \
  --output-dir external_validation
```

After generation, `synthetic-concordance` compares each observable construct
with the corresponding external-study partition.

## Code structure

- `external.sources` reads and verifies quantitative external data.
- `external.realism` compares source and generated distributions.
- `external.recovery` repeats mechanism and treatment-effect analyses.
- `trialeval` reconstructs TrialEval results from public release tables.
- `trialdev` reconstructs TrialDev observational and randomized-phase results.
- `validation_figures` renders the published plots from their CSV data.
- `io` provides shared JSON and checksum operations.

The top-level CLI groups complete workflows with focused analyses. Run
`trialagentbench-validate --help` to list the installed commands and
`trialagentbench-validate <command> --help` for required inputs. Commands that
construct a release analysis bundle operate on a new output directory and leave
source archives unchanged.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## Contributing

Open reproducible defects and focused pull requests in the
[TrialAgentBench repository](https://github.com/TrialAgentBench/TrialAgentBench-harness).
Validation changes should identify their source data and estimand, include
tests, and pass `python -m pytest`.

Report security-sensitive defects through GitHub private vulnerability
reporting. Do not place credentials, restricted source data, evaluator
contents, or exploitable details in a public issue.

## License

CC BY-NC 4.0. See [LICENSE](LICENSE).
