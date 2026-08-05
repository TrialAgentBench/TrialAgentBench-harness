# Reproduce the packaged analyses

Reproduction has three distinct targets: the software environment, the exact
benchmark archives, and the statistical result reconstructed from those
archives. Keeping them separate prevents a successful software test from being
mistaken for confirmation of a statistical conclusion.

## 1. Install the recorded environment

The repository is a two-package `uv` workspace supporting Python 3.11 to 3.13.
The root lockfile fixes the complete development and analysis environment.

```bash
uv sync --all-packages --all-extras
```

## 2. Match the analysis to its inputs

Data-specific reconstruction requires the corresponding participant, evaluator,
and verification archives from the
[TrialAgentBench dataset](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/).
The participant archive contains the task files shown to the agent. The
evaluator archive contains the analyses and decisions used for scoring. The
verification archive contains the independent reconstruction records. Only the
participant archive may be mounted into an agent workspace.

The validation package binds its methods, result tables, figures, source
records, and environment checksum in one inventory. Verify the installed bundle
before interpreting a result:

```bash
uv run --package trialagentbench-validation python -c \
  "from pathlib import Path; from trialagentbench_validation.contracts.simulation_validation_bundle import SimulationValidationBundleV1, verify_simulation_validation_bundle; root = Path('verification/validation_results'); bundle = SimulationValidationBundleV1.model_validate_json((root / 'validation_bundle.json').read_text()); verify_simulation_validation_bundle(root, bundle); print(bundle.checksum)"
```

## 3. Reconstruct the result

Choose the command reference that matches the task:

| Task | Command reference |
|---|---|
| Trace a reported conclusion to its evidence | [Verification guide](VERIFICATION_GUIDE.md) |
| Install the harness and run or grade an archive | [Harness quickstart](../harness/docs/QUICKSTART.md) |
| Reproduce a run identity, grade, or statistical reconstruction | [Harness reproducibility](../harness/docs/REPRODUCIBILITY.md) |
| Use a focused verification command or package API | [Verification package](../verification/README.md) |

Each reconstruction record retains the archive identifiers, input checksums,
method, estimate, uncertainty, and comparison tolerance needed to trace the
result back to the supplied data.

## 4. Check the software packages

```bash
make check
make test
make build
make smoke
```

The build creates wheels and source distributions for both packages. The smoke
test installs those wheels into a fresh virtual environment and checks both
command-line interfaces.
