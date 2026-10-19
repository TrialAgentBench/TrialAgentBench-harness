# Contributing

Thank you for helping improve TrialAgentBench. Please open focused issues and pull
requests that name the affected benchmark (TrialEval or TrialDev), command and
release.

## Development setup

The repository is a [uv](https://docs.astral.sh/uv/) workspace with three
distributions: the harness (`trial-agent-bench`, at the root), the independent
verifier (`verifier/`) and the clinical table projections (`clinical-data/`).
`uv.lock` pins the development dependencies. The three packages are not on PyPI;
install them together from the checkout. `SYNC.json` records the source snapshot
this tree was exported from and the digest of every file, so two exports of one
snapshot are byte-identical.

```bash
uv sync --all-packages --all-extras
uv run make check            # black, ruff, mypy and the offline test suites
uv run make test-executor    # Docker: build the executor and run its container tests
```

`make format` applies black and ruff fixes. Tests never contact a model provider;
tests that need Docker are marked `executor` and run only in `make test-executor`.

## Branches and pull requests

Open a pull request against `main` from a topic branch or fork. Describe the
problem, the resulting behavior and how you checked it. CI runs lint, types and
tests on every pull request; maintainers review changes before they are merged.

Each pull request should:

1. Add or update a deterministic test for the change.
2. Run `uv run make check`.
3. Never commit credentials, `.env` files, evaluator archives, model traces or run
   outputs (`results/`).
4. A scientific change states the clinical question, estimand, analysis route,
   assumptions and uncertainty rule it affects. A change to scoring, a reference or a
   released number updates its documentation in the same pull request.

## Security

Report security-sensitive problems, including any credential or evaluator answer
found in this repository, through GitHub private vulnerability reporting rather
than a public issue.
