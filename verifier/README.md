# TrialAgentBench Verifier

The verifier lets anyone check a TrialAgentBench result without trusting the harness that produced
it. It authenticates release downloads, replays submitted analyses in the pinned executor, and
recomputes numerical loss, decision regret, development value and campaign comparisons from the
recorded files. It independently implements the same **scorer** (versioned grading rules) and imports neither the harness nor the
simulator. Its replay uses the same isolated executor image as the harness.

The benchmark's methods are in
[METHODS.md](../docs/METHODS.md);
running an evaluation is covered in
[RUNNING.md](../docs/RUNNING.md).

Start with `practice` to check the installation, verify your release download, then use the
[smoke recheck](#worked-smoke-recheck) to reproduce your first campaign's grades.

## Install

With Python 3.11 or 3.12, from a clone of the [harness repository](https://github.com/TrialAgentBench/TrialAgentBench-harness),
`uv sync --locked` installs the verifier with the harness. Run the examples below from that checkout's
root with `uv run`; this selects the installed environment. To inspect the interface:

```bash
uv run trialagentbench-verify --help
```

To install the verifier alone:

```bash
python -m pip install ./verifier
```

It is not published on a package index; install it from the checkout or the release toolkit.
Commands that replay analyses (`execute`, `attest-executor`) need Docker; the others do not.
With a standalone pip installation, omit `uv run` from the commands below.

## Check the installation

```bash
uv run trialagentbench-verify practice --output practice-report.json --review-dir practice-review
```

This runs packaged known-answer cases through the scoring equations, without credentials, Docker or a
download, and writes `practice-review/index.html`. It checks the installation; it is not a model
evaluation.

## Verify a release download

Release-verification commands take expected digests, so a check binds to the release you intended rather than to
whatever files are present. Take digests from the release's checksum files: `SHA256SUMS` at the
dataset root covers every file, `full/SHA256SUMS` lists the catalogs and toolkit, and
`full/release-set-manifest.json.sha256` holds the release identity.
Obtain these anchors from the pinned release you trust. Checksums downloaded alongside files detect
corruption and mismatches; they do not independently establish who published those files.

The table gives syntax fragments; reporting commands also require `--output FILE`.

| Command | What it checks |
| --- | --- |
| `release-set DIR --expected-manifest-sha256 D` | A complete release directory: the manifest, the four catalogs, the toolkit and every shard |
| `release CATALOG --expected-sha256 D [--task-id T ...]` | One catalog's logical bytes and release qualification records, for all tasks or only the selected ones |
| `toolkit ZIP --expected-sha256 D` | The software toolkit archive and the distributions it ships |
| `assets inspect CATALOG --expected-sha256 D` | The size of a selection, without downloading payloads |
| `assets fetch CATALOG --expected-sha256 D --cache-directory C [--base-url URL]` | Acquires only the selected shards into a verified cache |
| `attest-executor TOOLKIT --expected-toolkit-sha256 D --executor-image IMAGE` | Builds the executor from an authenticated toolkit and records its environment hash |

For example, to verify a complete local copy, including all shards:

```bash
uv run trialagentbench-verify release-set benchmark/full \
  --expected-manifest-sha256 "$(cut -d' ' -f1 benchmark/full/release-set-manifest.json.sha256)" \
  --output release-set-report.json
```

A selected `release` check states which tasks it covered and makes no claim about omitted payloads.
Successful `release-set` output summarises verified artifacts and payload shards, byte counts,
available participant/evaluator roles and the limits of the check. It verifies integrity and declared
qualification records; it does not rerun scientific qualification or execute analyses. Qualification
checks are named **external anchor**, **table integrity** and **execution replay** in reports. The JSON
output retains the authenticated manifest.
The running guide's selective download deliberately omits shards, so it is insufficient for
`release-set` until every shard is present. To download a complete copy instead:

```bash
export HF_HOME="$PWD/cache/huggingface"
uvx --from huggingface_hub hf download TrialAgentBench/TrialAgentBench --repo-type dataset \
  --revision v1.0.1 --local-dir benchmark
(cd benchmark && sha256sum --check SHA256SUMS)
```

Missing, corrupt, unexpected or mismatched shard members fail before anything is materialised.
[RUNNING.md](../docs/RUNNING.md#2-download-and-verify-the-release)
describes selective acquisition and the cache.

## Re-score a campaign

`trialagentbench campaign finish` grades a campaign with the harness. The verifier recomputes the same
results from the campaign's recorded files, so a reader can check a published comparison
independently. Obtain the expected digests from whoever ran the campaign: they bind each check to the
intended plan, exports and release.

A TrialEval **series** is one clinical question and trial design; a **cell** applies an assumption
tier to it. A TrialDev **programme** is one simulated world shared by its budget and data-view
variants. Its two **streams** are initial evidence analysis and budgeted development; they are
compared separately.

In the current campaign implementation, TrialEval headlines use `primary` attempts (repetition one),
with an eligible `retry` replacing a primary after an evaluator error or pre-run identity exclusion
when the retry has no infrastructure fault. Later `stochastic_duplicate` attempts measure within-task
variability against the primary and do not enter headline estimates. TrialDev averages scheduled
repetitions and linked budget conditions within each world before headline means and paired
comparisons, separately by stream and data view. The frozen plan and attempt roles define the
aggregation; repeated runs do not increase the number of independent worlds. Model outcomes are
never retried.

Keep the whole campaign directory, not only its public summary: it retains the submitted scripts,
execution proofs and complete result exports. Paths below are relative to that directory.

| Command | What it recomputes | Inputs |
| --- | --- | --- |
| `analyse-trialeval` | TrialEval decision regret, consequential-error rate, numerical loss, completion and paired model comparisons | Campaign plan, result export, release identity and their digests |
| `analyse-trialdev` | TrialDev initial-analysis loss, acquired-study results and development value | Evaluator release, complete result export and the recorded invocation |
| `compare-trialdev` | Paired TrialDev model differences, with worlds as the resampling unit | Evaluator release, campaign plan and the index of every planned export |
| `behaviour` | Action and completion summaries and their figures | Trace bundle and its manifest digest |

Numerical **loss** combines normalised point error with interval width and missed coverage; lower
is better. **Bounded recovery** displays `1 / (1 + aggregate mean loss)` on the same panel with the
same weights and completeness rule; higher is better. TrialEval's headline is posterior-expected
decision regret in US$ millions (lower is better); TrialDev's is development value (higher is better).
[METHODS.md](../docs/METHODS.md#6-answers-and-grading) gives the equations and their limits.

Each command's `--help` lists its required paths and digests. Add `--review-dir NEW_DIRECTORY` to
`release`, `score`, `analyse-trialeval` or `analyse-trialdev` for an offline report with tables and
plots, and `--evidence-kind local_control`, `historical` or `live` to label the source of the results.

| Verification input | Campaign location |
| --- | --- |
| TrialEval combined plan | `comparisons/trialeval/plan.json` |
| TrialEval combined result export | The single ZIP in `comparisons/trialeval/export/` |
| TrialDev campaign plan | `campaign.json` |
| TrialDev export index | `comparisons/trialdev/exports.json`; its relative paths locate every job's ZIP |
| Per-job result exports and checks | `results/JOB_ID/export/` and `results/JOB_ID/verification/` |
| Release identity and analysis settings | `configuration.json` and `campaign.json` |

For a complete, uncensored TrialEval panel, run the following from the repository root. Set the
variables to the export path and independently retained expected identities; the command refuses
unset variables. `TRIALEVAL_PLAN_SHA256` is the SHA-256 of the plan **file**, not its JSON
`checksum` field. Use the campaign's recorded bootstrap settings if they differ from the defaults
(2,000 resamples, seed 65009, confidence 0.95):

```bash
uv run trialagentbench-verify analyse-trialeval \
  --campaign-plan results/comparison/comparisons/trialeval/plan.json \
  --expected-campaign-plan-sha256 "${TRIALEVAL_PLAN_SHA256:?trusted plan file SHA-256}" \
  --result-export "${TRIALEVAL_EXPORT:?path to the combined result ZIP}" \
  --expected-result-export-sha256 "${TRIALEVAL_EXPORT_SHA256:?trusted result ZIP SHA-256}" \
  --expected-release-id "${TRIALEVAL_RELEASE_ID:?trusted TrialEval release ID}" \
  --expected-release-manifest-sha256 "${TRIALEVAL_RELEASE_SHA256:?trusted TrialEval manifest SHA-256}" \
  --output results/trialeval-recheck.json
```

The TrialEval release ID and manifest digest here identify its logical release, not the enclosing
release-set manifest; the combined plan records both. An independent verifier should compare them
with the identities retained by the operator, rather than trusting a received plan alone.

For a complete, uncensored TrialDev panel, use the native plan and the complete export index.
Set both expected digests from the operator's independently retained records:

```bash
uv run trialagentbench-verify compare-trialdev benchmark/full/trialdev-evaluator.assets.json \
  --expected-release-sha256 "${TRIALDEV_CATALOG_SHA256:?trusted evaluator catalog SHA-256}" \
  --campaign-plan results/comparison/campaign.json \
  --expected-plan-sha256 "${CAMPAIGN_PLAN_SHA256:?trusted campaign plan file SHA-256}" \
  --exports results/comparison/comparisons/trialdev/exports.json \
  --output results/trialdev-recheck
```

### Worked smoke recheck

After the [documented smoke campaign](../docs/RUNNING.md#3-smoke-run) finishes, run this Bash
example from the repository root. `analyse-trialdev` checks one scheduled run; `compare-trialdev`
checks the whole smoke comparison. The invocation identity is the **recorded value** in
`campaign.json` → `runs[]` → `invocation_sha256`, not a digest to reconstruct from command arguments.
The export index also records each job's ZIP path and digest.

These commands print and use the actual local file digests and recorded invocation. For a recheck
of somebody else's campaign, compare these with independently retained identities from its operator
before running the verifier. The catalog digest below comes from the pinned dataset's checksum file.

```bash
set -euo pipefail
CAMPAIGN=results/smoke
CATALOG=benchmark/full/trialdev-evaluator.assets.json
CATALOG_SHA256=$(awk '$2 == "trialdev-evaluator.assets.json" {print $1}' benchmark/full/SHA256SUMS)
read -r JOB INVOCATION_SHA256 < <(uv run python - <<'PYTHON'
import json
from pathlib import Path
plan = json.loads(Path("results/smoke/campaign.json").read_text())
run = next(row for row in plan["runs"]
           if row["benchmark"] == "trialdev" and row["model_name"] == "smoke-zero-python")
print(run["job_id"], run["invocation_sha256"])
PYTHON
)
EXPORTS=("$CAMPAIGN/results/$JOB/export/"*.zip)
test "${#EXPORTS[@]}" -eq 1
EXPORT=${EXPORTS[0]}
EXPORT_SHA256=$(sha256sum "$EXPORT" | cut -d' ' -f1)
PLAN_SHA256=$(sha256sum "$CAMPAIGN/campaign.json" | cut -d' ' -f1)
printf 'job=%s\ninvocation_sha256=%s\nexport=%s\n' "$JOB" "$INVOCATION_SHA256" "$EXPORT"
sha256sum "$CATALOG" "$CAMPAIGN/campaign.json" "$EXPORT" "$CAMPAIGN/comparisons/trialdev/exports.json"

uv run trialagentbench-verify analyse-trialdev "$CATALOG" \
  --expected-release-sha256 "$CATALOG_SHA256" \
  --result-export "$EXPORT" --expected-result-export-sha256 "$EXPORT_SHA256" \
  --expected-invocation-sha256 "$INVOCATION_SHA256" \
  --output results/smoke-recheck/trialdev-analysis.json
uv run trialagentbench-verify compare-trialdev "$CATALOG" \
  --expected-release-sha256 "$CATALOG_SHA256" \
  --campaign-plan "$CAMPAIGN/campaign.json" --expected-plan-sha256 "$PLAN_SHA256" \
  --exports "$CAMPAIGN/comparisons/trialdev/exports.json" \
  --output results/smoke-recheck/trialdev-comparison
```

Both commands should exit zero. The per-run JSON retains the task grades and development value;
these are installation controls, not model results. In
`results/smoke-recheck/trialdev-comparison/analysis.json`, the `comparisons` and `leaderboard`
rows have `estimate: null`, null uncertainty and `unavailable_reason: "missing_required_stratum"`:
one world cannot represent all required programme strata. Leaderboard rows also carry
`uncertainty_unavailable_reason`. The smoke campaign's TrialEval public summary reports
`scientific_analysis_status: "unavailable_due_insufficient_independent_worlds"`, and its leaderboard
rows withhold estimates and uncertainty with `unavailable_reason: "insufficient_independent_worlds"`.
Per-task grades remain available under the campaign's `results/` tree.

For a budget-censored panel, both comparison commands accept
`--budget-censoring results/comparison/comparisons/censoring.json`. TrialEval also requires
`--campaign-index results/comparison/campaign.json` and
`--expected-campaign-index-sha256 "${CAMPAIGN_PLAN_SHA256:?trusted campaign plan file SHA-256}"`;
keep the original campaign tree so its frozen native plans can be authenticated. TrialDev binds
the censoring record to its `--campaign-plan`. These inputs reproduce the paired completed panel,
with censoring counts and rates alongside. A small diagnostic selection may have valid task grades
without supporting a comparison; missing strata, insufficient independent worlds or missing
scheduled outcomes withhold headline estimates and state why.

### One submission

Two commands check a single TrialEval or TrialDev initial analysis from outside a campaign. An analysis
project contains `analysis.py` or `analysis.R`, its supporting modules and `estimates.csv`, optionally
under `scratch/`:

```csv
target_id,estimate,lower,upper
```

| Command | What it checks | Inputs |
| --- | --- | --- |
| `execute` | That the project reproduces its numbers in two fresh executions in the pinned executor | Participant release and digest, task ID, project directory, executor image ID and environment hash |
| `score` | Numerical loss, bound to the executed analysis | Evaluator release and digest, the CSV (as `--submission` and `--program-result`), the script (`--program`), and the execution proof from `execute` with its SHA-256 |

Participant files alone do not contain the references needed for grading. For a TrialDev development
episode, `execute` stages only the initial evidence; the harness captures later readouts and replays
each submitted analysis against the evidence available at that time, and `analyse-trialdev` checks
those retained records without rerunning them.

## Reading a failure

A verifier failure means a check failed, not that an estimate was merely inaccurate; the JSON report
names the failed check. A missing numerical answer has unavailable loss and appears in completion
counts. File integrity, reproducibility and scientific recovery answer different questions, and none
establishes clinical effectiveness outside the simulated populations.

Archive inspection and extraction share finite rejection limits: 4 million members, 32 GiB
compressed, 64 GiB uncompressed and 2 GiB per member, with structured inputs and checksum inventories
limited to 1 GiB. These are refusal limits, not memory estimates; a complete campaign export can be
large, so use explicit task subsets on smaller machines.

[Citation](CITATION.cff) · [Licence](LICENSE)

Scored artifacts carry a scorer rule version and a SHA-256 of the installed scientific
scoring sources of both distributions; the harness verifies its declared source digest before grading.
Analysis and public-result readers require the current identity;
comparisons and leaderboard exports refuse mixed identities. Package versions alone
cannot identify a scorer. To update retained campaign results without agent requests,
use `trialagentbench campaign regrade --campaign results/original --output results/regraded`
in the harness. It preserves original custody and records the previous and current scorer
in the new root's `regrade.json` before applying the ordinary offline finalizer.
