# TrialAgentBench Verifier

The verifier makes TrialAgentBench's scientific comparisons and scores independently
inspectable. Its reader reconstructs trial outcomes, compares analyses of the same
clinical question, and follows study decisions through benefit, harm and expenditure.
Its execution commands reproduce submitted code and recalculate numerical loss and
development value. Neither route imports Ariadne or the harness; the harness shares
the isolated executor, while the verifier implements the scoring equations separately.

## Review the science

The [scientific walkthrough](examples/scientific/README.md) connects trial
construction, clinical records, treatment-effect recovery and development decisions.
It includes data, equations, quantitative figures and R comparisons. Reproduce it
from the public source repository:

```bash
uv sync --extra analysis
uv run python verifier/examples/scientific/review.py --compact --output scientific-review
```

Open `scientific-review/index.html`. The compact walkthrough follows an actual
three-study rheumatoid-arthritis programme through its readouts, time and spending,
and reproduces the bundled Parkinson analysis with conventional statistical libraries.

The [complete scientific review](examples/scientific/README.md) adds the teaching
study atlas, mechanism experiments, interpretation materiality and paired research
strategies. Obtain its record archives, omit `--compact`, and open the generated
`index.html`. Each figure has a numerical table. The public reader separates routine
measurements from sampled adjudication and preserves pairing between related studies.

The commands below check submitted agent outputs against the specified quantities.

## Check an installation

```bash
trialagentbench-verify practice --output practice-report.json --review-dir practice-review
```

This runs packaged arithmetic examples without credentials or Docker. Open
`practice-review/index.html` for the results. It is an installation check, not a
model evaluation.

## Verify results

Normal evaluations use `trialagentbench campaign finish` to reproduce analyses,
grade them and assemble comparisons. Return the complete campaign directory to
retain reports, code, decisions and resource use. The standalone commands support
independent inspection and reproduction:

| Command | What it checks | Inputs |
| --- | --- | --- |
| `release` | File integrity and applicable reference checks | Release archive and its expected digest |
| `execute` | Reproduction of submitted numbers in two fresh Docker executions | Participant archive, task, analysis project and executor identity |
| `score` | Numerical loss and its connection to the executed analysis | Evaluator archive, results, analysis project and execution proof |
| `analyse-trialeval` | Mean loss, completion and paired comparisons | Campaign plan and result export |
| `analyse-trialdev` | Initial-analysis loss, acquired-study results and development value | Evaluator archive, complete result export and recorded invocation |
| `compare-trialdev` | Paired model differences with worlds as the resampling unit | Campaign plan and complete job export index |
| `behaviour` | Action and completion summaries | Trace bundle |
| `toolkit`, `release-set`, `attest-executor` | Matching distributions, archives and container environment | Toolkit or release files and expected identities |

Each command's `--help` gives its required paths and checksums. Add
`--review-dir NEW_DIRECTORY` to `release`, `score`, `analyse-trialeval` or
`analyse-trialdev` for an offline report with tables and applicable plots.
Campaign reports also take `--evidence-kind local_control`, `historical` or `live`
to identify the source of the results. Keep that label consistent with the run.

An analysis project contains `analysis.py` or `analysis.R`, `estimates.csv` and any
supporting modules, optionally under `scratch/`. The CSV has four columns:

```csv
target_id,estimate,lower,upper
```

For TrialEval and TrialDev initial analysis, `score` takes this CSV as both
`--submission` and `--program-result`, and the executed `analysis.py` or `analysis.R`
as `--program`. Its `--execution-proof` comes from `execute` or the benchmark runner,
with that proof's SHA-256 as `--expected-execution-proof-sha256`. Obtain expected
release and proof digests from the evaluator, so the checks bind the result to the
intended files and execution.
Participant files alone do not contain the reference answers required for grading.

Standalone `execute` stages the initial task evidence. TrialDev's runner captures
later study readouts and reproduces each submitted analysis against the evidence
available at that time. `analyse-trialdev` checks those retained records and
recalculates results; it does not rerun the analyses. Use the harness's regrading
command when fresh execution of a saved development run is needed.

Missing numerical answers retain unavailable loss and appear in completion counts.
A verifier failure means a check failed, not that a model's estimate was merely
inaccurate. The JSON report identifies the failed check. File integrity,
reproducibility and scientific recovery answer different questions; none establishes
clinical effectiveness outside the simulated populations.

Use `release EVALUATOR.zip --development --expected-sha256 DIGEST --output REPORT.json`
to check a development archive before running it. The report checks file integrity
and role identities and explicitly records that scientific qualification was not
assessed. `analyse-trialdev --development` also supports results from these archives,
retaining the usual scoring checks.

Archive inspection and extraction share finite limits: 4 million members,
32 GiB compressed, 64 GiB uncompressed and 2 GiB per member. Structured input
files and checksum inventories are limited to 1 GiB. Full-custody exports include
both original and graded snapshots; even a single raw-panel control exceeds
100,000 members. These are rejection limits, not memory-use estimates or a promise
that every campaign fits on every host. Retain sufficient storage for authenticated
copies and extraction, and use explicit task subsets for smaller machines.

[Citation](CITATION.cff) · [License](LICENSE)
