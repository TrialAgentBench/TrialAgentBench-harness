from __future__ import annotations

from pathlib import Path


def test_no_payload_key_probing_in_offline_core() -> None:
    """Guardrail: do not treat `.payload[...]` as a hidden contract dependency."""
    root = Path(__file__).resolve().parents[1] / "trialagentbench_harness"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        # Wrappers are the only place allowed to translate raw payload dicts into
        # schema-bearing contracts.
        if path.name == "grade_wrappers.py":
            continue
        text = path.read_text(encoding="utf-8")
        if ".payload[" in text or "payload[" in text:
            offenders.append(str(path))
    assert not offenders, f"Found payload key-probing in offline core: {offenders}"
