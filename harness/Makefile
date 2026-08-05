.PHONY: format lint typecheck test test-executor build check

PY ?= uv run python
DEV_PY ?= uv run --extra dev python

format:
	$(DEV_PY) -m black .

lint:
	RUFF_NO_CACHE=true $(DEV_PY) -m ruff check . --fix

typecheck:
	tmpdir=$$(mktemp -d); \
	  MYPY_CACHE_DIR=$$tmpdir $(DEV_PY) -m mypy \
	    trialagentbench_harness; \
	  rc=$$?; rm -rf "$$tmpdir"; exit $$rc

test:
	$(DEV_PY) -m pytest -q -p no:cacheprovider -c pyproject.toml --rootdir=.

test-executor:
	docker build -t trialagentbench/executor:0.1.0 executor
	$(DEV_PY) -m pytest -q -p no:cacheprovider -c pyproject.toml --rootdir=. -m executor tests/execution

build:
	$(DEV_PY) -m build

check: lint typecheck test
