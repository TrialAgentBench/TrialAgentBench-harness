# Changelog

All notable changes to the TrialAgentBench harness, verifier, executor and documentation.

## [1.0.1] - 2026-10-25

A software release. Tasks, worlds, data archives, catalogs, selections and grading keys are those
of 1.0.0, byte for byte; the dataset is published at revision `v1.0.1`. Results graded by the 1.0.0
harness carry no scorer identity; regrade them with `campaign regrade`, which reruns no agent,
before comparing them with 1.0.1 results.

### Added

- `trialagentbench campaign regrade --output DIR` finishes a retained campaign's frozen submissions
  again, offline, into a new directory under the installed scorer; the original campaign is not
  changed. `--release DIR` grades against another release set that serves byte-identical campaign
  inputs in the same executor environment.
- Scorer provenance: grades, analyses, leaderboards, exports and verification reports record the
  declared scoring rule and the SHA-256 of the harness and verifier scoring sources. Comparisons,
  exports and published results refuse inputs graded by another scorer.

### Changed

- Numerical target loss is computed exactly and reported to 12 significant digits, ties to even,
  identically in harness grading and the verifier, so last-digit floating-point differences no
  longer reach scores.
- Bounded recovery is `1 / (1 + L)` of the aggregate primary numerical loss `L`, with its interval
  and a delta-method standard error, and is withheld when `L` is. It previously averaged per-task
  transforms with a missing answer scored 0. Rankings by recovery now always agree with rankings by
  loss.
- Campaign analyses and leaderboards withhold model comparisons and uncertainty when a required
  stratum has fewer than two independent worlds; task grades are kept. Their result files have new
  schema versions.
- The analysis container is tagged `trialagentbench/executor:1.0.1`; its sources and environment
  are those of 1.0.0. Rebuild it under the new tag.
- Verification reports name verifier 1.0.1; execution proofs written by verifier 1.0.0 still verify.
- `certifi>=2026.10.19` is a runtime dependency, for its certificate-store security fixes.

### Fixed

- Independent replay authenticates the same archive-backed evidence the harness stages, including
  its ZIP aliases.
- The executor test suite matches the current agent-turn and provider-response contracts.
- The site's product tests run before the Pages deployment, and its screenshots render reliably.
  Its build uses Astro 7.3.8, with patched Vite, Zod and undici, clearing every `npm audit` finding.
- Documentation: the overview and score vocabulary, task units, the verifier workflow, demonstration
  prerequisites and interpretation, regrading, and a short provenance record for the bundled demo.

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
