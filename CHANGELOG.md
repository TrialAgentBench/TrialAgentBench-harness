# Changelog

All notable changes to the TrialAgentBench harness, verifier, executor and documentation.

## [1.0.0] - 2026-10-20

The first release of TrialAgentBench.

### Benchmarks

- **TrialEval**: 17 trial series and 21 regime cells on three assumption tiers (12 A1, five A2 twins
  and four A3 traps; which series carry the twins and traps is not published, though in the public
  split task counts show which series hold two cells), over 68 independent base worlds. Each trial is presented in five
  evidence contexts (C1–C5): 420 public tasks, of which the 84 C5 tasks form the main selection. A
  separately seeded withheld split of 68 base worlds presents one cell per law in C5 (68 tasks),
  including one held-out trap cell; it is held and scored by the leaderboard operator and is not
  distributed.
- **TrialDev**: 384 public programmes in rheumatoid arthritis, migraine prevention, atopic dermatitis
  and COPD exacerbations, each with three or five candidate treatments, a constrained, lean and
  generous budget and a six-year horizon: 3,072 tasks, 768 in the main selection and 256 in the
  ablation subset. A separately seeded withheld release of 64 programmes is held by the operator.

### Scoring

- TrialEval's headline is the posterior-expected regret, in US$ millions, of the decision each answer
  drives (stop, or confirm at a chosen size) under the disclosed effect prior and declared per-trial
  economics, against the closest credited analysis, repriced across a ×0.5/×2 economics band, with
  the consequential-error rate (a changed stop-or-confirm decision costing at least $5M). Numerical
  point and interval loss is secondary; plan-execution accuracy is scored apart in C1 and C3.
- TrialDev reports initial and acquired-study numerical loss and risk-adjusted development value
  against stopping, fixed controls, an evidence-responsive reference and an attainable oracle.
- Aggregate score estimates carry bootstrap standard errors over independent worlds when the panel
  supports inference, with a small-sample correction within each resampling stratum; paired between-model
  contrasts report Holm-adjusted p-values and minimal detectable differences (an unadjusted normal approximation).

### Software

- `trialagentbench` harness: campaign preview, prepare, run, status, resume, retry and finish;
  selective, checksum-verified acquisition of task shards; offline Python and R execution in the
  pinned `trialagentbench/executor:1.0.0` container; replay of every saved analysis before grading.
- `trialagentbench-verify`: independent release, toolkit and executor verification, submission
  replay and scoring, and campaign re-analysis without importing the harness.
- Documentation: [README](README.md), [RUNNING](docs/RUNNING.md), [METHODS](docs/METHODS.md) and the
  [verifier guide](verifier/README.md).
