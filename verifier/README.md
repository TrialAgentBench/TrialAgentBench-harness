# TrialAgentBench Verifier

The verifier lets anyone check a TrialAgentBench result without trusting the harness that produced
it. It authenticates release downloads, replays submitted analyses in the pinned executor, and
recomputes numerical loss, decision regret, development value and campaign comparisons from the
recorded files. It implements the scoring equations itself and imports neither the harness nor the
simulator; the two share only the isolated executor image.

The benchmark's methods are in
[METHODS.md](https://github.com/TrialAgentBench/TrialAgentBench-harness/blob/main/docs/METHODS.md);
running an evaluation is covered in
[RUNNING.md](https://github.com/TrialAgentBench/TrialAgentBench-harness/blob/main/docs/RUNNING.md).

## Install

From a clone of the [harness repository](https://github.com/TrialAgentBench/TrialAgentBench-harness),
`uv sync` installs the verifier with the harness and provides the `trialagentbench-verify` command.
To install the verifier alone:

```bash
python -m pip install ./verifier
```

It is not published on a package index; install it from the checkout or the release toolkit.
Commands that replay analyses (`execute`, `attest-executor`) need Docker; the others do not.

## Check the installation

```bash
trialagentbench-verify practice --output practice-report.json --review-dir practice-review
```

This runs packaged known-answer cases through the scoring equations, without credentials, Docker or a
download, and writes `practice-review/index.html`. It checks the installation; it is not a model
evaluation.

## Verify a release download

Every command takes an expected digest, so a check binds to the release you intended rather than to
whatever files are present. Take digests from the release's checksum files: `SHA256SUMS` at the
dataset root covers every file, `full/SHA256SUMS` lists the catalogs and toolkit, and
`full/release-set-manifest.json.sha256` holds the release identity.

| Command | What it checks |
| --- | --- |
| `release-set DIR --expected-manifest-sha256 D` | A complete release directory: the manifest, the four catalogs, the toolkit and every shard |
| `release CATALOG --expected-sha256 D [--task-id T ...]` | One catalog's logical bytes and release qualification records, for all tasks or only the selected ones |
| `toolkit ZIP --expected-sha256 D` | The software toolkit archive and the distributions it ships |
| `assets inspect CATALOG --expected-sha256 D` | The size of a selection, without downloading payloads |
| `assets fetch CATALOG --expected-sha256 D --cache-directory C [--base-url URL]` | Acquires only the selected shards into a verified cache |
| `attest-executor TOOLKIT --expected-toolkit-sha256 D --executor-image IMAGE` | Builds the executor from an authenticated toolkit and records its environment hash |

For example, to verify the whole downloaded release:

```bash
trialagentbench-verify release-set benchmark/full \
  --expected-manifest-sha256 "$(cut -d' ' -f1 benchmark/full/release-set-manifest.json.sha256)" \
  --output release-set-report.json
```

A selected `release` check states which tasks it covered and makes no claim about omitted payloads.
Missing, corrupt, unexpected or mismatched shard members fail before anything is materialised.
[RUNNING.md](https://github.com/TrialAgentBench/TrialAgentBench-harness/blob/main/docs/RUNNING.md#2-download-and-verify-the-release)
describes selective acquisition and the cache.

## Re-score a campaign

`trialagentbench campaign finish` grades a campaign with the harness. The verifier recomputes the same
results from the campaign's recorded files, so a reader can check a published comparison
independently. Obtain the expected digests from whoever ran the campaign: they bind each check to the
intended plan, exports and release.

| Command | What it recomputes | Inputs |
| --- | --- | --- |
| `analyse-trialeval` | TrialEval decision regret, consequential-error rate, numerical loss, completion and paired model comparisons | Campaign plan, result export, release identity and their digests |
| `analyse-trialdev` | TrialDev initial-analysis loss, acquired-study results and development value | Evaluator release, complete result export and the recorded invocation |
| `compare-trialdev` | Paired TrialDev model differences, with worlds as the resampling unit | Evaluator release, campaign plan and the index of every planned export |
| `behaviour` | Action and completion summaries and their figures | Trace bundle and its manifest digest |

Each command's `--help` lists its required paths and digests. Add `--review-dir NEW_DIRECTORY` to
`release`, `score`, `analyse-trialeval` or `analyse-trialdev` for an offline report with tables and
plots, and `--evidence-kind local_control`, `historical` or `live` to label the source of the results.

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
names the failed check. A missing numerical answer keeps unavailable loss and appears in completion
counts. File integrity, reproducibility and scientific recovery answer different questions, and none
establishes clinical effectiveness outside the simulated populations.

Archive inspection and extraction share finite rejection limits: 4 million members, 32 GiB
compressed, 64 GiB uncompressed and 2 GiB per member, with structured inputs and checksum inventories
limited to 1 GiB. These are refusal limits, not memory estimates; a complete campaign export can be
large, so use explicit task subsets on smaller machines.

[Citation](CITATION.cff) · [Licence](LICENSE)
