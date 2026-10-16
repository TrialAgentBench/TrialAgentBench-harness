# Investigate a clinical world

These executable demonstrations connect the benchmark’s clinical mechanisms to
trial records, statistical interpretations and research decisions. They answer
four questions: what world produced the evidence, what the records reveal, what
an analysis recovers, and whether its interpretation changes the conclusion.

## Start with a research episode

From the public repository root:

```bash
uv sync --extra analysis
uv run python verifier/examples/scientific/review.py --compact --output compact-review
```

Open `compact-review/index.html`. The report follows an executed
rheumatoid-arthritis programme through studies with 20, 120 and 600 patients.
Their readouts arrive on programme days 150, 400 and 810. The analyses estimate
treatment-minus-control DAS28 differences at 28, 84 and 168 days of follow-up,
respectively. Lower disease activity favours treatment.

Opening a study activates sites and incurs setup charges. Recruitment and
follow-up determine when outcomes become available. Readouts change the evidence;
continuing research consumes time and money. The report joins native action
requests and ledger receipts to independently recomputed clinical estimates.
The programme is scripted, making the world’s responses reproducible.

At submission, 740 patients have entered the three studies. Direct study charges
are $30.13 million and programme overhead is $12.15 million. The $50 million
research budget covers study charges and outstanding patient obligations;
overhead separately reduces terminal value. The remaining research budget is
$19.87 million.

| Output | What it explains |
| --- | --- |
| `research-episode.svg` | Studies, observed effects and spending on one programme clock |
| `interaction-timeline.csv` | Each study start, readout and terminal decision with its recorded response |
| `trialdev_readout_estimates.csv` | Clinical and safety estimates, uncertainty and horizon-specific references |
| `trialdev_study.md` | Patient records, ordinary-library analyses and the terminal assessment |
| `trialdev_resources.csv` | Study charges from the recorded tariffs and enrolments |
| `trialdev_value_components.csv` | Treatment benefit, harm, studies and overhead in programme value |

Compact mode also analyses the bundled Parkinson trial and observational
starting cohort. It runs without Docker, model calls or network access. Use a new
output directory each time; `--source /path/to/scientific-data` selects a separate
copy with the same input layout.

## Reproduce the complete scientific account

The complete account also reads the six teaching-record archives listed in
`SHA256SUMS`, which source checkouts and distributions omit. Download them from
the dataset's [`scientific-evidence/`](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/tree/main/scientific-evidence)
directory into this directory and check them with
`sha256sum -c SHA256SUMS`, then run:

```bash
uv run python verifier/examples/scientific/review.py --output scientific-review
```

Open `scientific-review/index.html`. The full report adds current trial patterns,
controlled mechanism experiments, interpretation materiality and paired research
strategies. Each display has exact numerical tables and editable Python source;
figures also have PDF, PNG and narrow-layout SVG versions where needed.

| Scientific question | Calculation and outputs |
| --- | --- |
| How does the world define a reference? | `known_law.csv` propagates disease states and treatment response; `trialdev_initial_reference_derivation.csv` integrates the observational starting-cohort associations. |
| Do trial designs appear in their records? | `current-trials/` measures the current study archive; `environment/design_evidence.csv` contains the separate worked rollout, monitoring and observation examples. |
| Do analyses recover the specified target? | `repeated_study_estimates.csv` and `recovery_operating_characteristics.csv` report effects, bias and interval coverage across repeated trials. |
| When does an interpretation change? | `interpretation-materiality.csv` classifies paired analyses under one fixed rule; `interpretation-materiality-pairs.csv` retains every changed and unchanged conclusion. |
| Which research choices are consequential? | `authored_programme_values.csv` and `authored_study_estimates.csv` retain all paired programmes and their readouts. |
| Does the conclusion depend on research value or prices? | `authored_utility_comparisons.csv` revalues the recorded programmes; actions and evidence remain fixed. |

Current-trial characterization reads `current-trial-records.zip`. Older bundles
without that archive still reproduce their scientific examples, but their report
does not present those examples as the current benchmark census. Each independent
study counts once; multiple information conditions are views of that same study.

The materiality experiment compares routine response labels and stratified
central adjudication in 32 matched colorectal trials. Both methods address the
same Day-168 tumour-response risk difference. A 95% interval entirely above a
required gain supports that target; an interval below it rules the target out;
otherwise the conclusion remains unresolved. The plotted required gains run
from zero to ten percentage points. These are interpretation scenarios, not
clinical recommendations or simulated funding decisions.

At a five-percentage-point target, routine analysis rules out the gain in 14/32
trials; adjudication does so in 0/32. Eighteen paired conclusions differ. The
separate complete-outcome reference is 7.50 percentage points (95% Monte Carlo
interval 7.13–7.87). Improved recovery need not establish a decisive conclusion
in an individual trial: in the first archived study, adjudication leaves the
five-point target unresolved.

## Follow an analysis through grading

```bash
uv run python verifier/examples/scientific/review_adjudication.py --output colorectal-review
```

This command refits both methods in all 32 colorectal trials and traces the first
archived trial through the public scorer. Its target is combination-minus-standard
therapy objective tumour response at Day 168. Both fits use 3,196 randomized
patients; 954 have sampled central adjudication.

Routine labels estimate −0.19 percentage points (95% interval −3.47 to 3.09).
Stratified adjudication estimates +5.15 points (−0.82 to 11.12). False-positive
control responses and missed active-arm responses explain the change; the
comparison holds the trial and clinical question fixed.

`adjudication-worked.svg` displays estimates and uncertainty.
`adjudication-grading.svg` decomposes numerical loss into estimation error,
interval width and a missed-reference penalty. Their wider interval costs the
adjudicated analysis more, but avoids the routine interval’s large miss penalty.
`worked-study-estimates.csv`, `worked-study-strata.csv` and
`observation-transitions.csv` expose the calculations. Different methods are
assessed by their answers, not their names.

Across these 32 trials, routine bias is −5.49 percentage points and coverage is
2/32; adjudication bias is +0.10 points and coverage is 31/32. The 64 separate
complete-outcome studies supply the reference and its uncertainty. The grading
example uses their contrast standard deviation as its normalization scale.

## Compare research strategies

The full report reconstructs six worlds, five designs and two allocation repeats
per design: 60 pairs of fixed-continuation and observed-evidence stopping policies.
`programme-decisions.svg` explains two selected pairs through their readouts,
decisions, costs and time. `authored-policy-panel.svg` displays every pair.

The selected effective and harmful worlds use 20-, 120- and 900-person studies.
These paired paths serve a different purpose from the compact 20/120/600 episode:
they explain how the same strategy can have different consequences across worlds.
Within each world, the policies share evidence until their actions diverge.
Later studies estimate later follow-up outcomes, not repeated estimates of the
early effect. The separate 60-person `trialagentbench demo run` fixture introduces
the actual agent tools and executable workspace.

To redraw the programme figures or revalue already reconstructed results:

```bash
uv run python verifier/examples/scientific/review_designs.py \
  --programmes scientific-review/authored_programme_values.csv \
  --preferences scientific-review/authored_utility_preferences.csv \
  --output programme-figures
```

This holds actions, approval and timing fixed. It measures sensitivity to treatment
value and resource prices; it does not rerun adaptive decisions.

## Scientific scope

Clinical reports inform selected patient distributions and AACT registry summaries
inform selected conduct parameters. Compare compatible clinical quantities;
registry duration is not patient follow-up, and noncompletion is not endpoint
unavailability. Larger simulated participant counts support controlled evaluation;
participant-count resemblance is not the main realism question.

Current trial characterization, repeated mechanism experiments and external data
answer different questions. The historical family examples contain 30 design and
assumption combinations from 17 clinical families. Recovery experiments have 32
trials per condition and separate reference panels; some conditions reuse the same
latent outcomes. The development experiment’s two allocation repeats share each
fixed clinical world. Those dependencies remain explicit in the calculations.

The [statistician guide](../../../docs/STATISTICAL_REVIEW.md) explains targets,
construction and assessment. The [clinical guide](../../../docs/GUIDE.md) follows
the research experience. Reconstruction and numerical checks are available with
the scientific outputs; they support reproduction rather than establish clinical
or decision transfer.

## Independent calculations and external records

With R and its `survival` package installed:

```bash
uv run python verifier/examples/scientific/review.py --output scientific-review-r --rscript Rscript
```

This adds R planning and survival calculations. Python-only execution does not
imply these optional comparisons ran.

The separate external experiment uses real RCT Bench records:

```bash
git clone https://github.com/bingkaiwang/rct_bench-main.git ../rct-bench
git -C ../rct-bench checkout --detach ef11941dc4749c135c71a1171c7ace70d7b7ca1c
uv run python verifier/examples/scientific/review_external.py --source ../rct-bench --output external-review
```

It recalculates adjusted contrasts in 33 trials and the paired resampling
experiment in ten trials. These are observed-data comparisons, not known clinical
population effects or a test of agent decisions. The separate
[decision-transfer protocol](DECISION_TRANSFER.md) specifies an unexecuted study
using independent programme dossiers.
