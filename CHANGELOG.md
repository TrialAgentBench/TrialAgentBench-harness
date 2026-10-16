# Changelog

All notable changes to the TrialAgentBench harness, verifier and documentation.

## [1.0.1] - 2026-10-16

The benchmark data, task identifiers, references and scores are unchanged from 1.0.0.

- Licences: the software (harness, verifier, clinical-data tables and executor) is
  Apache-2.0, except the executor's bundled R bridge package, which keeps its MIT licence;
  the Hugging Face dataset is CC BY 4.0. Both were CC BY-NC 4.0.
- The `main` and `withheld` campaign panels now select TrialDev's documented main
  selection from its configuration manifest: 768 public and 128 withheld tasks. The
  1.0.0 panels scheduled every raw TrialDev task (1,536 and 256). The manifests ship as
  `main/trialdev-configuration.json` and `withheld/trialdev-configuration.json`, and
  the 32-programme ablation subset as `trialdev-ablation/` (256 tasks).
- `docs/CAMPAIGN_CONFIGURATIONS.md` lists every campaign setting, its values and
  default, whether it is part of a run's scored condition, and the selection sizes.

## [1.0.0] - 2026-10-16

### Benchmark

- TrialEval is rebuilt as 21 regime cells: an A1 cell in each of the 17 series,
  where the prespecified analysis is valid under realistic background conduct,
  and four A3 consequential traps (TE-S06-A3, TE-S08-A3, TE-S15-A3, TE-S16-A3).
  A2 and A4 are defined but unpopulated. Each series has four laws, giving 68
  independent base worlds; a series' A1 and A3 cells share each base world. Each
  world and cell is presented in five evidence contexts (C1–C5): 420 tasks, of
  which the 84 C5 tasks form the main selection. The construct is frozen for
  this release.
- Trials match the median randomized enrolment of matched AACT trials, except the
  trap series TE-S06 (9,504), TE-S08 (114,000) and TE-S15 (660), each sized at the
  minimum where its trap is consequential at confirmatory, power-planned size;
  TE-S07 (7,680; power-planned against four registry trials); and TE-S10 (52, the
  registry floor).
- Each TrialEval trial draws its true effect from a disclosed prior: the
  signal-to-noise mixture fitted to 23,551 Cochrane randomised trials. The
  TrialEval headline is decision consequence: each answer drives a stop-or-confirm
  decision under that prior, and leaderboards report mean regret in US$ millions
  and the consequential-error rate. Numerical loss is secondary; bounded recovery
  no longer ranks TrialEval.
- TrialDev is regenerated as 384 programmes (128 strata) in four clinical
  programmes (rheumatoid arthritis, migraine prevention, atopic dermatitis, COPD),
  with the candidate prior recentred (20% effective, 50% ineffective). Each has an
  initial analysis and three development budgets, 0.75, 1.1 and 2.5 times the
  programme's lean-programme cost (replacing the fixed $50M-$400M ladder), each in raw
  and analysis-ready form (3,072 tasks). The reference policy pools the declared
  prior, the observed association (with a confounding allowance) and the
  randomized readout. A configuration manifest selects the main
  run at run time: each programme once, at one budget rotated within strata
  (768 tasks), plus a 32-programme ablation subset with every budget and view.
- Separately seeded withheld splits are scored privately. TrialEval's holds 68
  independent base worlds (four laws per series, none shared with the public
  split) presented only in C5: 88 tasks over the public cells plus one held-out
  trap cell. TrialDev's holds 64 programmes (512 tasks).
- Both TrialEval splits are qualified with no failed cell, and both regenerated
  TrialDev splits are qualified with no failed world.
  TrialEval recovery precision uses 1,200 prior-drawn and 1,600 coverage worlds
  per cell.
- Decision regret uses declared per-trial economics: confirmatory cost per
  participant at 0.6 times the published average for the trial type, a fixed cost
  of $5.9M (industry) or $1.0M (routine-data public health), and a value of
  success from median ten-year sales (or a declared public-health value), so that
  value ranges from $588.7M to
  $1,552.1M across cells.
- Where the analysis plan is visible (C1, C3), TrialEval tasks also request a
  `prespecified_primary_analysis` row: the plan's primary analysis executed exactly
  as written. It is accurate when the point and both interval endpoints lie within
  0.05 normalization SD of the plan route's own result on the comparison scale; a
  missing or unreproduced row is not accurate. It is scored apart from numerical
  loss and the decision, checked by the verifier as `score.plan_execution`, and
  reported per model in public campaign results as `plan_execution_summaries`.
- The harness identity is no longer bound to qualification; a harness edit needs
  no requalification.

### Agent environment

- TrialDev programmes start from a committee brief and use flat programme tools
  (`start_study`, `request_readout`, `adapt_study`, `wait_until`,
  `stop_development`, `advance_candidate`).
- TrialDev allows 100 agent turns per submission step; TrialEval keeps 50 turns
  per task.
- Tasks left unrun by an exhausted campaign spending limit are reported as
  budget-censored and excluded from paired comparisons.

### Repository

- New standalone layout: the harness at the repository root with `verifier/`,
  `clinical-data/`, `docs/`, `examples/` and `executor/`, managed as one uv
  workspace with a lock matching the qualified host versions. It replaces the
  earlier `harness/` and `verification/` layout.

## 2026-08-05

- First public release of the harness and independent verifier.
