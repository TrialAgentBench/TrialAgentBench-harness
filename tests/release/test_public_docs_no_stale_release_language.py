"""Check the consumer route without enforcing prose or documentation counts."""

import re
from pathlib import Path


def test_public_entry_and_guide_links_resolve() -> None:
    """Installed-package source links must reach real current consumer inputs."""
    root = Path(__file__).resolve().parents[2]
    guides = (root / "README.md", *(root / "docs").glob("*.md"))
    for guide in guides:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", guide.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            destination = (guide.parent / target.split("#", 1)[0]).resolve()
            # The public mirror nests the separately distributed verifier package.
            if not (root / "verifier").exists() and destination.is_relative_to(root / "verifier"):
                destination = root.parent / "verifier" / destination.relative_to(root / "verifier")
            assert destination.is_file(), (guide, target)


def test_current_guides_do_not_restore_superseded_grading_rules() -> None:
    """Reject concrete obsolete contracts that would change participant behaviour."""
    root = Path(__file__).resolve().parents[2]
    text = " ".join(
        " ".join(path.read_text().lower().split()) for path in (root / "README.md", *(root / "docs").glob("*.md"))
    )
    for obsolete in (
        "the primary programme-level loss is the final decision's expected opportunity loss",
        "requires an uncertainty-qualified identified analysis",
        "a narrower interval cannot support nomination",
        "reward qualified sensitivity or deferral, not recovery of hidden state",
        "structured submission interface",
    ):
        assert obsolete not in text


def test_consumer_guides_carry_the_training_corpus_canary() -> None:
    """The documented canary must equal the one stamped into catalogs and shards."""
    from trialagentbench_verifier.assets import TRAINING_CANARY

    root = Path(__file__).resolve().parents[2]
    for guide in (root / "docs" / "QUICKSTART.md", root / "docs" / "BENCHMARK_AND_DATA.md"):
        assert TRAINING_CANARY in " ".join(guide.read_text(encoding="utf-8").split()), guide
