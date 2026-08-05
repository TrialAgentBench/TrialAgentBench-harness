from __future__ import annotations

import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path


def test_distributions_contain_public_runtime_and_reproducibility_assets(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output), str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(output.glob("*.whl"))
    sdists = tuple(output.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())

    prefix = "trialagentbench_harness/resources"
    assert f"{prefix}/agent_output_schema.json" in members
    assert f"{prefix}/eval_spec.json" in members
    assert {
        f"{prefix}/examples/submissions/trialeval_shapes.json",
        f"{prefix}/examples/submissions/trialdev_observational_review.json",
        f"{prefix}/examples/submissions/trialdev_phase_request.json",
        f"{prefix}/examples/submissions/trialdev_phase_analysis.json",
        f"{prefix}/examples/submissions/trialdev_phase_decision.json",
    } <= members
    assert all(member.startswith(("trialagentbench_harness/", "trial_agent_bench-")) for member in members)

    with tarfile.open(sdists[0]) as archive:
        source_members = {Path(name).name for name in archive.getnames()}
    assert {"README.md", "LICENSE", "Makefile"} <= source_members


def test_package_metadata_points_to_canonical_public_repositories() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["license"] == "CC-BY-NC-4.0"
    assert project["authors"] == [{"name": "Anonymous Authors"}]
    assert project["urls"] == {
        "Homepage": "https://github.com/TrialAgentBench/TrialAgentBench-harness",
        "Repository": "https://github.com/TrialAgentBench/TrialAgentBench-harness",
        "Data": "https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/",
        "Documentation": "https://github.com/TrialAgentBench/TrialAgentBench-harness/tree/main/harness/docs",
        "Issues": "https://github.com/TrialAgentBench/TrialAgentBench-harness/issues",
    }
