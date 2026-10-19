# Changelog

All notable changes to the TrialAgentBench harness, verifier, executor and documentation.

## [1.0.0] - 2026-10-19

The first release of TrialAgentBench.

### Benchmarks

- **TrialEval**: 17 trial series and 21 regime cells on three assumption tiers (12 A1; five A2 twins,
  TE-S04-A2, TE-S06-A2, TE-S15-A2, TE-S16-A2 and TE-S17-A2; four A3 traps, TE-S06-A3, TE-S14-A3,
  TE-S15-A3 and TE-S16-A3), over 68 independent base worlds. Each trial is presented in five
  evidence contexts (C1–C5): 420 public tasks, of which the 84 C5 tasks form the main selection. A
  separately seeded withheld split of 68 base worlds presents one cell per law in C5 (68 tasks),
  including one held-out trap cell; it is held and scored by the leaderboard operator and is not
  distributed.
- **TrialDev**: 384 public programmes in rheumatoid arthritis, migraine prevention, atopic dermatitis
  and COPD exacerbations, each with three or five candidate treatments, a constrained, lean and
  generous budget and a six-year horizon: 3,072 tasks, 768 in the main selection and 256 in the
  ablation subset. A separately seeded withheld release of 64 programmes is held by the operator.

### Scoring

- TrialEval's headline is the posterior-expected regret, in US$ millions, of the stop-or-confirm
  decision each answer drives under the disclosed effect prior and declared per-trial economics,
  against the closest credited analysis, with the consequential-error rate (a changed decision costing
  at least $5M) and a ×0.5/×2 economics band. Numerical point and interval loss is secondary;
  plan-execution accuracy is scored apart in C1 and C3.
- TrialDev reports initial and acquired-study numerical loss and risk-adjusted development value
  against stopping, fixed controls, an evidence-responsive reference and an attainable oracle.

### Software

- `trialagentbench` harness: campaign preview, prepare, run, status, resume, retry and finish;
  selective, checksum-verified acquisition of task shards; offline Python and R execution in the
  pinned `trialagentbench/executor:1.0.0` container; replay of every saved analysis before grading.
- `trialagentbench-verify`: independent release, toolkit and executor verification, submission
  replay and scoring, and campaign re-analysis without importing the harness.
- Documentation: [README](README.md), [METHODS](docs/METHODS.md), [RUNNING](docs/RUNNING.md) and the
  [verifier guide](verifier/README.md).
