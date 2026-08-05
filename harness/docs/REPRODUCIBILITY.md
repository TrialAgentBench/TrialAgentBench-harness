# Reproduce runs, grades, and statistical results

Reproduction has three distinct targets. Keeping them separate prevents a
successful software replay from being mistaken for evidence about statistical
performance.

## 1. Establish the exact inputs

The participant archive contains the task files shown to the agent, the
evaluator archive contains the analyses and decisions used for scoring, and the
verification archive contains the independent reconstruction records.

Before running or grading:

1. verify each archive against the release checksum file;
2. retain the release identifier, source commit, environment-lock checksum, and
   experiment-design checksum;
3. extract participant, evaluator, and verification archives into separate
   directories; and
4. reject duplicate, absolute, parent-traversing, linked, or case-colliding
   archive members.

The participant directory is the only directory that may be mounted into a
model workspace.

## 2. Replay a run and its grade

Save every run identity, assignment identity, model snapshot, experiment
condition, request-replicate identifier, task-materialization seed, optional
provider decoding seed, assistance condition, interface condition, resource
limit, completion state, and raw response. When a provider-specific reasoning
control is used, retain its checksum-bound capability record. Grade saved
submissions offline against the exact evaluator release used for the run.

```bash
trialagentbench run --help
trialagentbench grade --help
trialagentbench export results --help
```

Missing, failed, and invalid assignments remain in the prespecified
denominator. Re-running only successful assignments changes the estimand and is
not a reproduction.

## 3. Reconstruct the statistical result independently

Three terms describe distinct operations:

- **archive verification** checks file membership, identity, and checksums;
- **result reconstruction** recomputes a released answer from
  participant-visible data; and
- **simulation validation** measures outcome replication, structural fidelity,
  and recovery of known generating effects.

The separately implemented `trialagentbench-validation` package distributed with the
[TrialAgentBench dataset](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/)
reconstructs declared results from participant-visible inputs before opening
evaluator output. For TrialEval analysis-ready items it
reports the route, comparison rule, numerical difference, declared tolerance,
and difference-to-tolerance ratio. Nonnumeric routes use exact code membership.

For every C5 item, verification:

1. identifies the affected domain using the public integrity policy;
2. identifies the unique exact duplicate using canonical typed row keys and
   row payloads;
3. removes exactly the declared duplicate count;
4. establishes canonical-content equality with the matched C4 item; and
5. only then compares the reconstructed analysis result with evaluator records.

The release is invalid if any required item is missing, unsupported, or outside
its declared tolerance.

The complete release contains one canonical submission per released TrialEval
item and TrialDev evaluation lane. The validation package
freezes its own grade projection first, then invokes the separately installed
public harness and compares exact records:

```bash
trialagentbench-validate grader-concordance \
  --release-root <extracted-release> \
  --canonical-submissions <extracted-release>/canonical_submissions \
  --output-dir <new-concordance-output>
```

The command fails on an incomplete denominator, unsupported record, subprocess
failure, or any field mismatch. The verifier runs its grade projection in a
separate process before invoking the public grader.

### Simulation results

The validation distribution includes `validation_results/REPORT.md`.
The report embeds the headline survival, ordinal, longitudinal,
joint-structure, effect-recovery, and control displays beside plain-language
interpretations. Each display links to its CSV data and analysis note, and the
package bundle binds every file to the verifier lock.

```python
from trialagentbench_validation.external.release.bundle import (
    installed_validation_root,
    verify_installed_validation_bundle,
)

bundle = verify_installed_validation_bundle()
print(installed_validation_root() / bundle.report.relative_path)
```

## Dependence and uncertainty

The 500 TrialEval items are not 500 independent trials. Five context views share
each generated base trial. Single-asset TrialDev checkpoints and objective
views share one programme trajectory. Portfolio objective and budget views
share one of 12 evidence worlds, and checkpoints share the same immutable world
history. Analyses therefore use the release-declared independence unit and
matched-set identity. Publication intervals use the frozen crossed-cluster
contracts and seeds in `RELEASE_MANIFEST.json`; users should not treat context
views, objective or budget views, checkpoints, or alternative scoring routes as
independent observations.

## Released worlds

The published world establishes complete construction, role separation,
deterministic replay, and workflow execution. The declared independent
generation set estimates operating characteristics through repeated-world
checks. Comparisons retain release identity and use the declared independence
units.

## Reproduction record

Retain:

- release and archive checksums;
- environment and source identities;
- immutable schedule and analysis configuration;
- API transport, gateway or pinned upstream provider, and exact model snapshot
  identities;
- experiment conditions and request-replicate identifiers;
- procedure-assistance level, tool-choice policy, per-step turn ceiling,
  corrected-submission limit, materialisation-retry limit, and any
  reported-cost stop record with its scheduled-unit partition;
- task-materialization, optional provider-decoding, and analysis seeds;
- raw and normalized submissions;
- complete grade records, including typed failures;
- independent verification receipts; and
- result-export manifest and checksum.

These records are sufficient to distinguish artifact reproduction, software
replay, independent statistical reconstruction, and repeated-world
performance evaluation.

Export completed run, grade, and verification roots without modifying the
release:

```bash
trialagentbench export results \
  --release-root <extracted-release> \
  --run-root <run-output> \
  --grade-root <grade-output> \
  --verification-root <verification-output> \
  --output-dir <new-result-bundle-directory>
```

The output contains one deterministic ZIP, its detached SHA-256 checksum, and
a receipt that binds every included file to the release identity. Result
artifacts and the output directory must remain outside the immutable release.
