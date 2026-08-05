"""Record an installed-wheel replay of an external qualification report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

DECLARED_NUMERIC_TOLERANCE = 1e-9


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReplayWheelV1(_FrozenModel):
    """Identity of the independently installed validation wheel."""

    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReplayReleaseV1(_FrozenModel):
    """Immutable identities carried by a qualification report."""

    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReplayComparisonV1(_FrozenModel):
    """Numerical and structural comparison with the reference report."""

    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structural_or_non_numeric_differences: int = Field(ge=0)
    maximum_absolute_numeric_difference: float = Field(ge=0, allow_inf_nan=False)
    declared_numeric_tolerance: float = Field(
        default=DECLARED_NUMERIC_TOLERANCE,
        gt=0,
        allow_inf_nan=False,
    )


class ReplayIsolationV1(_FrozenModel):
    """Runtime isolation facts reported by the clean interpreter."""

    python: str
    validation_module_path: str
    repository_on_import_path: bool
    repository_modules_imported: tuple[str, ...]


class QualificationCleanReplayV1(_FrozenModel):
    """Complete clean-wheel replay record."""

    schema_id: str = "trialagentbench.qualification_clean_wheel_replay/v1"
    wheel: ReplayWheelV1
    release_identity: ReplayReleaseV1
    comparison: ReplayComparisonV1
    isolation: ReplayIsolationV1
    scientific_dependencies: dict[str, str]


def record_clean_replay(
    *,
    reference_report: Path,
    replay_report: Path,
    wheel: Path,
    clean_python: Path,
    repository_root: Path,
    output: Path,
) -> QualificationCleanReplayV1:
    """Compare qualification reports and attest to clean interpreter isolation."""

    reference = json.loads(reference_report.read_text(encoding="utf-8"))
    replay = json.loads(replay_report.read_text(encoding="utf-8"))
    differences, maximum_difference = _compare_payloads(reference, replay)
    if differences:
        raise ValueError(
            f"Clean replay differs structurally or in non-numeric values at {differences} locations."
        )
    if maximum_difference > DECLARED_NUMERIC_TOLERANCE:
        raise ValueError(
            "Clean replay exceeds the declared numeric tolerance: "
            f"{maximum_difference:.6g} > {DECLARED_NUMERIC_TOLERANCE:.6g}."
        )
    runtime = _clean_runtime(clean_python, repository_root=repository_root)
    module_path = Path(runtime["validation_module_path"]).resolve()
    repository = repository_root.resolve()
    repository_on_import_path = module_path.is_relative_to(repository)
    repository_modules = tuple(
        str(value) for value in runtime["repository_modules_imported"]
    )
    if repository_on_import_path or repository_modules:
        raise ValueError(
            "Clean replay imported repository code outside the installed validation wheel."
        )
    module_location = "/".join(module_path.parts[-2:])
    report = QualificationCleanReplayV1(
        wheel=ReplayWheelV1(filename=wheel.name, sha256=_sha256_file(wheel)),
        release_identity=ReplayReleaseV1(
            design_sha256=str(reference["design_sha256"]),
            receipt_sha256=str(reference["receipt_sha256"]),
        ),
        comparison=ReplayComparisonV1(
            reference_sha256=_sha256_file(reference_report),
            wheel_replay_sha256=_sha256_file(replay_report),
            structural_or_non_numeric_differences=differences,
            maximum_absolute_numeric_difference=maximum_difference,
            declared_numeric_tolerance=DECLARED_NUMERIC_TOLERANCE,
        ),
        isolation=ReplayIsolationV1(
            python=str(runtime["python"]),
            validation_module_path=module_location,
            repository_on_import_path=repository_on_import_path,
            repository_modules_imported=repository_modules,
        ),
        scientific_dependencies={
            str(key): str(value)
            for key, value in dict(runtime["scientific_dependencies"]).items()
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _compare_payloads(reference: object, replay: object) -> tuple[int, float]:
    differences = 0
    maximum = 0.0

    def visit(left: object, right: object) -> None:
        nonlocal differences, maximum
        if type(left) is not type(right):
            differences += 1
            return
        if isinstance(left, dict):
            right_mapping = cast(dict[object, object], right)
            if left.keys() != right_mapping.keys():
                differences += 1
                return
            for key in left:
                visit(left[key], right_mapping[key])
            return
        if isinstance(left, list):
            right_list = cast(list[object], right)
            if len(left) != len(right_list):
                differences += 1
                return
            for left_item, right_item in zip(left, right_list, strict=True):
                visit(left_item, right_item)
            return
        if isinstance(left, float):
            maximum = max(maximum, abs(left - cast(float, right)))
        elif left != right:
            differences += 1

    visit(reference, replay)
    return differences, maximum


def _clean_runtime(clean_python: Path, *, repository_root: Path) -> dict[str, Any]:
    code = """
import json
import pathlib
import sys
import numpy
import pandas
import pyarrow
import pydantic
import scipy
import statsmodels
import lifelines
import trialagentbench_validation

repository = pathlib.Path(sys.argv[1]).resolve()
repository_modules = sorted(
    name
    for name, module in sys.modules.items()
    if getattr(module, "__file__", None)
    and pathlib.Path(module.__file__).resolve().is_relative_to(repository)
)
print(json.dumps({
    "python": sys.version.split()[0],
    "validation_module_path": str(pathlib.Path(trialagentbench_validation.__file__).resolve()),
    "repository_modules_imported": repository_modules,
    "scientific_dependencies": {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "pydantic": pydantic.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "lifelines": lifelines.__version__,
    },
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(clean_python), "-c", code, str(repository_root.resolve())],
        check=True,
        capture_output=True,
        text=True,
        cwd="/tmp",
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Record a clean-wheel qualification replay."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--clean-python", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record_clean_replay(
        reference_report=args.reference_report,
        replay_report=args.replay_report,
        wheel=args.wheel,
        clean_python=args.clean_python,
        repository_root=args.repository_root,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
