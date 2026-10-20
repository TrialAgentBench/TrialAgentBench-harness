"""Check the consumer route without enforcing prose or documentation counts."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_GUIDES = (_ROOT / "README.md", *sorted((_ROOT / "docs").glob("*.md")))


def test_public_entry_and_guide_links_resolve() -> None:
    """Installed-package source links must reach real current consumer inputs."""
    for guide in _GUIDES:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", guide.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            destination = (guide.parent / target.split("#", 1)[0]).resolve()
            # The public mirror nests the separately distributed verifier package.
            if not (_ROOT / "verifier").exists() and destination.is_relative_to(_ROOT / "verifier"):
                destination = _ROOT.parent / "verifier" / destination.relative_to(_ROOT / "verifier")
            assert destination.is_file(), (guide, target)


def test_consumer_guides_carry_the_training_corpus_canary() -> None:
    """The documented canary must equal the one stamped into catalogs and shards."""
    from trialagentbench_verifier.assets import TRAINING_CANARY

    for guide in (_ROOT / "docs" / "RUNNING.md", _ROOT / "docs" / "METHODS.md"):
        assert TRAINING_CANARY in " ".join(guide.read_text(encoding="utf-8").split()), guide
