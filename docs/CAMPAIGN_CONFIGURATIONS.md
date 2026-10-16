# Campaign configurations

This page lists every setting of a TrialAgentBench campaign: what it accepts, its default, what
it changes for the agent, and whether it is part of a run's scored condition. Use it to choose a
comparison before writing `campaign.json`. Settings that are part of the scored condition must be
identical across the models you compare; operational settings only change how a run is executed.

## What can be selected

Both benchmarks are released as a public split, which you score yourself, and a withheld split,
whose evaluator is held privately.

| Benchmark | Split | Units | Tasks |
| --- | --- | --- | ---: |
| TrialEval | public | 17 trial series, 21 regime cells (17 A1 plus 4 A3 traps), 68 independent base worlds, 84 trials, each in five evidence contexts C1–C5 | 420 |
| TrialEval | withheld | the 21 public cells plus one held-out trap cell, 68 base worlds, C5 only | 88 |
| TrialDev | public | 384 programmes, each an independent world with its own seed: for each of 4 identification regimes and 2 portfolio sizes (3 or 5 candidates), 48 worlds covering 4 diseases × 4 environment profiles × 3 seeds; per programme 1 evidence analysis and 3 budgeted portfolio episodes, each raw or analysis-ready | 3,072 |
| TrialDev | withheld | 64 programmes | 512 |

The release ships these task selections, each with a `TASKS.csv` and a `campaign-panel.json` to
copy into `panel`:

| Selection | TrialEval | TrialDev | Tasks per model and repetition |
| --- | ---: | ---: | ---: |
| `main/` (headline comparison) | 84 (C5) | 768 | 852 |
| `withheld/` (scored privately) | 88 (C5) | 128 | 216 |
| `trialdev-ablation/` (paired budget and view contrasts) | – | 256 | 256 |
| `full/` (every task) | 420 | 3,072 | 3,492 |

TrialDev's main selection runs each programme once: its raw evidence analysis and one raw
portfolio episode, with the budget rotated within strata over three rungs at 0.75, 1.1 and 2.5
times the programme's lean cost (128 episodes per rung). `main/trialdev-configuration.json` lists
it and the 32-programme ablation subset, which runs all eight tasks of each programme, one programme per
stratum. The toolkit grades ablation tasks individually; it does not compute budget or view contrasts for them,
so paired contrasts from this subset are estimates without intervals unless you add your own analysis.

You can also select any subset by task ID. TrialEval task IDs fix the evidence context; TrialDev
selectors may be a task ID, a programme world (`tdw-…`), a stream (`evidence_analysis`,
`portfolio_management`) or `world:stream`. A leaderboard needs a complete selection: TrialEval's
is the C5 tasks of every cell, and TrialDev's needs every stratum with at least two worlds.
Smaller subsets give per-task grades only.

### TrialEval task factors

These are fixed by each task, so you choose them by choosing tasks.

| Factor | Values | Notes |
| --- | --- | --- |
| Evidence context | C1 analysis-ready data with a locked analysis plan; C2 analysis-ready, protocol only; C3 raw domains with a locked plan; C4 raw domains, protocol only; C5 raw domains with a declared data defect, protocol only | C5 is the headline. C1 and C3 also score execution of the plan's primary analysis. |
| Assumption tier | A1 (the prespecified analysis is valid under realistic conduct), A3 (a consequential trap) | A2 and A4 are defined but have no cells in this release. Each A3 cell carries one trap over its series' realistic background conduct; traps are not stacked. |
| Design family | individual randomised, pragmatic, covariate structure, endpoint ascertainment, cluster parallel, stepped wedge, group sequential | Reported as descriptive families. |
| Law | four effect laws per series, each drawn from the disclosed effect prior | A base world is one series and law; all its contexts share it. |

### TrialDev task factors

| Factor | Values |
| --- | --- |
| Stream | evidence analysis (no budget); portfolio management (budgeted, sequential studies and a stop-or-advance decision) |
| Data view | raw records; analysis-ready tables (scored as a separate track) |
| Budget rung | 0.75×, 1.1× or 2.5× the programme's lean cost |
| Programme | rheumatoid arthritis, migraine prevention, atopic dermatitis or COPD exacerbations; 3 or 5 candidates; four identification regimes |

## Release and input settings

| Setting | Values | Default | Effect |
| --- | --- | --- | --- |
| `release_directory` | path to the downloaded `full/` directory | required | The catalogs, shards and toolkit the campaign uses. |
| `expected_release_manifest_sha256` | the release-set manifest digest | read from `release-set-manifest.json.sha256` | Refuses a release whose manifest differs. |
| `input_mode` | `qualified`, `development` | `qualified` | `development` runs unqualified development archives and cannot be combined with the withheld split. |
| `executor_image_id` | a Docker image ID | built and attested at preparation | Must match the manifest's executor environment hash. |

## Scored condition settings

Each of these enters the run's recorded condition. Compare models only within one combination;
a different value is a different condition.

| Setting | Values | Default | Effect |
| --- | --- | --- | --- |
| `panel.split` | `public`, `withheld` | `public` | Which archives are run; withheld needs the private evaluator. |
| `panel.analysis_language` | `python_r`, `python`, `r` | `python_r` | Interpreters the agent may use. |
| `panel.procedure_assistance` | `output_contract_only`, `unordered_checklist` | `output_contract_only` | Adds an unordered checklist to TrialEval's instructions. TrialEval only: a panel containing TrialDev must use `output_contract_only`. |
| `panel.trialeval_turns` | integer ≥ 1 | 50 | Turn limit per TrialEval task. |
| `panel.trialdev_turns_per_step` | integer ≥ 1 | 100 | Turn limit per TrialDev submission step. |
| `panel.watchdog_seconds` | integer ≥ 1 | 3,600 (TrialEval), 1,800 (TrialDev) | Wall time per task; one value applies to both benchmarks. |
| `panel.repetitions` | integer ≥ 1 | 1 | Independent requests per task, recorded as separate replicates. |
| `panel.condition_id` | label | `primary` | Names the condition; results compare only within one label. |

### Per-model settings (`models[]`)

| Setting | Values | Default | Notes |
| --- | --- | --- | --- |
| `name` | a unique label | required | Names the model's results. |
| `provider` | `openrouter`, `offline_control` | `openrouter` | `offline_control` runs the scripted Python and R controls (`smoke-zero-python`, `smoke-zero-r`) with no model and no reasoning settings. |
| `capability_snapshot` | path to a saved route snapshot | fetched from OpenRouter at preparation | Records the route's supported parameters, prices and limits in the run's condition. |
| `model`, `upstream_provider` | an OpenRouter model ID and its exact upstream provider | required | Requests are pinned to that provider with no fallback. Web-search routes are refused. Each model ID may appear once per campaign, so compare reasoning efforts or providers in separate campaigns. |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or omitted | provider default | Must be supported by the route. |
| `exclude_reasoning`, `replay_reasoning` | true or false | false, false | Not both true; replay requires OpenRouter. |
| `maximum_completion_tokens` | integer ≥ 1 | 32,768 | Must not exceed the route's limit. |
| `send_temperature`, `decoding_seed` | true or false; integer or null | false, null | By default no temperature is sent, so the provider's default sampling applies. |
| `request_timeout_seconds`, `retry` | seconds; attempts and delay | 90 s; 5 attempts, 2 s | Provider requests only. |

## Fixed settings

These are the same in every campaign and record their identity in each run: bounded conversation
memory; tool choice `auto`; a 120,000-character context limit; the analysis container
`trialagentbench/executor:1.1.1` (environment hash `296477ba…`) with no network, 2 CPUs, 4,096 MB
and 300 s per code execution; the tool sets (code execution, Parquet inspection and analysis checks,
workspace files, submission; for TrialDev also study actions and the stop-or-advance decision); and
the release's prompt, scorer and harness source hashes.

## Operational settings

These change how a run is executed, not what is measured: `operations.model_workers`,
`task_workers`, `executor_workers` (at least twice their product), `grading_workers`, campaign and
per-model `spend_limit_micro_usd`, request reservations and pricing, `requests_per_minute` and
`tokens_per_minute` (equal for models sharing a provider route), and the analysis bootstrap
repetitions and seed. Every value, including these, is bound into the campaign plan, so changing
one starts a new campaign. Tasks left unrun when a spending limit is reached are reported as
censored, and the analysis warns when censoring differs between models by more than five points.

## Invalid combinations

- A withheld split with `input_mode: development`.
- `unordered_checklist` with TrialDev in the panel.
- Both `exclude_reasoning` and `replay_reasoning`; a reasoning effort or token limit the route does
  not support; reasoning settings on the offline control.
- The same model ID twice in one campaign; different rate limits for models sharing a route.
- Analysis language, assistance, turns or watchdog varying within a campaign: they are panel-wide.

## Size of the configuration space

Holding each model's settings fixed, the scored conditions are split × language × assistance:
2 × 3 × 2 = 12 for TrialEval and 2 × 3 = 6 for TrialDev. Each model setting multiplies these
(up to eight reasoning efforts, three reasoning-return modes, temperature sent or not, seed set or
not), and turns, watchdog and token limits are continuous. Crossing the benchmark conditions with
the released tasks gives 13,800 task-conditions per model and repetition: TrialEval public
420 × 6 and withheld 88 × 6; TrialDev public 3,072 × 3 and withheld 512 × 3.

The recommended comparison holds every setting at its default and runs `main/` once per model:
852 tasks per model (84 TrialEval, 768 TrialDev), each with a complete trace of tool calls,
provider requests and costs for failure review.
