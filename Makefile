.PHONY: install check test build smoke clean

SMOKE_PYTHON ?= 3.11

install:
	uv sync --all-packages --all-extras

check:
	uv run --directory harness ruff check .
	uv run --directory harness black --check .
	uv run --directory harness mypy trialagentbench_harness
	uv run --directory verification ruff check .
	uv run --directory verification mypy src/trialagentbench_validation

test:
	uv run --directory harness pytest
	uv run --directory verification pytest

build:
	uv build --package trial-agent-bench --out-dir dist/harness
	uv build --package trialagentbench-validation --out-dir dist/verification

smoke: build
	uv venv --clear --python $(SMOKE_PYTHON) .smoke-venv
	uv pip install --python .smoke-venv/bin/python dist/harness/*.whl dist/verification/*.whl
	.smoke-venv/bin/trialagentbench --help
	.smoke-venv/bin/trialagentbench-validate --help

clean:
	rm -rf .smoke-venv dist
