"""Tests for the public documentation contract."""

from __future__ import annotations

from pathlib import Path

from trialagentbench_harness.contracts.trace.bundle import OBSERVABLE_TRACE_TABLE_MODELS

PUBLIC_GUIDES = (
    "README.md",
    "docs/BENCHMARK_AND_DATA.md",
    "docs/CONTRACTS.md",
    "docs/EXPERIMENTS.md",
    "docs/QUICKSTART.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SCORING.md",
)

STALE_DOC_PATTERNS = (
    "Python 3.11",
    "3.11+",
    "judged by an eval-side LLM",
    "provider default",
    "backward compat",
    "legacy",
    "xlarge",
    "six-model",
    "Publication reproduction starts",
    "publication reproduction should start",
)

FIXED_PANEL_DOC_PATTERNS = (
    "GPT-5.4",
    "named model panel",
    "Gemini 3.1",
    "Qwen",
    "Kimi",
    "GLM",
    "Luna",
    "gpt-5_4",
    "model_slug",
    "paper traces",
    "current trace examples",
    "exactly the same numbers",
)

NONPUBLIC_CONSTRUCTION_PATTERNS = (
    "Conventional starting analysis",
    "TE-S01",
    "multiplied by 1.35",
    "no progression, phase 1 failure",
)


def _harness_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "trialagentbench_harness").is_dir():
            return parent
    raise RuntimeError("Could not locate harness root")


def _public_docs(root: Path) -> list[str]:
    return list(PUBLIC_GUIDES)


def test_public_documentation_surface_is_exact() -> None:
    """The source repository must contain only the supported public guides."""

    root = _harness_root()
    actual = {"README.md", *(path.relative_to(root).as_posix() for path in (root / "docs").rglob("*.md"))}
    assert actual == set(PUBLIC_GUIDES)


def test_public_docs_do_not_contain_stale_release_language() -> None:
    """Public docs must describe the current publication release contract."""
    root = _harness_root()
    offenders: list[str] = []
    for relative in _public_docs(root):
        text = (root / relative).read_text(encoding="utf-8")
        for pattern in (
            *STALE_DOC_PATTERNS,
            *FIXED_PANEL_DOC_PATTERNS,
            *NONPUBLIC_CONSTRUCTION_PATTERNS,
        ):
            if pattern.casefold() in text.casefold():
                offenders.append(f"{relative}: {pattern}")
    assert not offenders, "\n".join(offenders)


def test_public_docs_make_execution_and_evidence_contract_explicit() -> None:
    """Public docs must state the execution and evidence contract."""

    root = _harness_root()

    required_fragments = {
        "README.md": (
            "The benchmark accepts multiple correct answers",
            "Only participant-role files enter the agent workspace",
        ),
        "docs/CONTRACTS.md": (
            "completed user-owned runs",
            "model names, counts, and ordering are not fixed",
            "does not infer private reasoning",
            "seven schema-backed tables",
        ),
    }

    missing: list[str] = []
    for relative, fragments in required_fragments.items():
        text = " ".join((root / relative).read_text(encoding="utf-8").split())
        for fragment in fragments:
            if fragment not in text:
                missing.append(f"{relative}: {fragment}")

    assert not missing, "\n".join(missing)


def test_contract_guide_lists_exact_public_trace_bundle() -> None:
    """The contract guide must describe outputs emitted by the public builder."""

    text = (_harness_root() / "docs" / "CONTRACTS.md").read_text(encoding="utf-8")

    assert set(OBSERVABLE_TRACE_TABLE_MODELS) == {
        "action_events.csv",
        "evidence_use.csv",
        "failure_cascades.csv",
        "semantic_features.csv",
        "trialdev_phase_outcomes.csv",
        "trialdev_program_cascades.csv",
        "unit_features.csv",
    }
    assert all(f"`{table_name}`" in text for table_name in OBSERVABLE_TRACE_TABLE_MODELS)
    assert "trace_bundle_manifest.json" in text
    assert "trialagentbench verify analysis-bundle" not in text
    assert "conclusion_derivation_catalog.csv" not in text
    assert "trace_to_score_linkage.csv" not in text


def test_public_guide_allowlist_is_exact() -> None:
    """The exported harness has one coherent seven-guide documentation surface."""

    assert set(_public_docs(_harness_root())) == set(PUBLIC_GUIDES)


def test_benchmark_guide_matches_executable_assumption_and_context_semantics() -> None:
    """Public factor definitions must retain the canonical scoring distinctions."""

    text = " ".join((_harness_root() / "docs" / "BENCHMARK_AND_DATA.md").read_text(encoding="utf-8").split())

    assert "makes the default route materially misleading" in text
    assert "C1 and C3 prescribe exactly one accepted route" in text
    assert "C2, C4, and C5 expose the complete accepted same-estimand route set" in text
    assert "Endpoints, competing events, and safety outcomes retain their own declared roles" in text
    assert "Death can occupy different roles for different questions" in text
