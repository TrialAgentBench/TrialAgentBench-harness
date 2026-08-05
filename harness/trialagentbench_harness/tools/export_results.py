"""Build a deterministic, checksummed archive of caller-owned results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_harness.contracts.release.result_export import (
    ReleaseStage,
    ResultArtifactKind,
    ResultExportBundleManifestV1,
    ResultExportMemberV1,
    ResultExportSourceV1,
)

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_FORBIDDEN_DIRECTORY_NAMES = frozenset({".git", ".mypy_cache", ".pytest_cache", "__pycache__"})
_FORBIDDEN_FILE_NAMES = frozenset({".env", "credentials.json", "provider.env"})
_FORBIDDEN_FILE_SUFFIXES = (".key", ".pem", ".pyc", ".pyo")
_RESULT_ARTIFACT_KINDS: tuple[ResultArtifactKind, ...] = (
    "run",
    "grade",
    "verification",
    "analysis",
)


class _ReleaseArtifactIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ReleaseManifestProjectionV1(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    release_id: str = Field(min_length=1)
    release_stage: ReleaseStage
    result_export_manifest: _ReleaseArtifactIdentityV1


class _ResultExportContractV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.export_results_manifest/v1"]
    release_id: str = Field(min_length=1)
    status: Literal["awaiting_model_runs"]
    canonical_result_schema_path: str = Field(min_length=1)
    simulation_properties_path: str = Field(min_length=1)
    expected_analysis_unit_count: int = Field(gt=0)
    join_key: Literal["canonical_result_join_key"]
    denominator_policy: Literal["retain_every_scheduled_assignment_with_typed_status"]
    result_export_command: Literal["trialagentbench export results"]
    required_artifact_kinds: tuple[Literal["run", "grade", "verification"], ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialagentbench export results",
        description=(
            "Archive immutable run, grade, and independent-verification outputs "
            "against one released benchmark identity."
        ),
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--grade-root", type=Path, action="append", required=True)
    parser.add_argument("--verification-root", type=Path, action="append", required=True)
    parser.add_argument("--analysis-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_release_member(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ValueError("release artifact paths must use POSIX separators")
    relative = PurePosixPath(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("release artifact paths must be safe relative paths")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
        raise ValueError(f"release artifact is missing or unsafe: {relative_path}")
    return target


def _release_contract(root: Path) -> tuple[_ReleaseManifestProjectionV1, str]:
    release_manifest_path = root / "RELEASE_MANIFEST.json"
    if not release_manifest_path.is_file() or release_manifest_path.is_symlink():
        raise ValueError("release root contains no regular RELEASE_MANIFEST.json")
    release_manifest = _ReleaseManifestProjectionV1.model_validate_json(
        release_manifest_path.read_text(encoding="utf-8")
    )
    contract_path = _canonical_release_member(root, release_manifest.result_export_manifest.path)
    if _sha256_file(contract_path) != release_manifest.result_export_manifest.sha256:
        raise ValueError("result-export contract checksum does not match RELEASE_MANIFEST.json")
    contract = _ResultExportContractV1.model_validate_json(contract_path.read_text(encoding="utf-8"))
    if contract.release_id != release_manifest.release_id:
        raise ValueError("result-export contract identifies a different release")
    if contract.required_artifact_kinds != ("run", "grade", "verification"):
        raise ValueError("result-export contract has an unsupported required-artifact policy")
    _canonical_release_member(root, contract.canonical_result_schema_path)
    _canonical_release_member(root, contract.simulation_properties_path)
    return release_manifest, _sha256_file(contract_path)


def _safe_label(path: Path) -> str:
    label = re.sub(r"[^a-z0-9_-]+", "-", path.name.casefold()).strip("-_")
    if not label:
        raise ValueError(f"artifact root has no usable label: {path}")
    return label[:64]


def _artifact_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"result artifact root must be a regular directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if any(part in _FORBIDDEN_DIRECTORY_NAMES for part in relative.parts[:-1]):
            raise ValueError(f"result artifact contains a transient directory: {relative.as_posix()}")
        if path.is_symlink():
            raise ValueError(f"result artifact contains a symbolic link: {relative.as_posix()}")
        mode = path.stat().st_mode
        if path.is_dir():
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"result artifact contains a non-regular file: {relative.as_posix()}")
        if path.name.casefold() in _FORBIDDEN_FILE_NAMES or path.suffix.casefold() in _FORBIDDEN_FILE_SUFFIXES:
            raise ValueError(
                f"result artifact contains a forbidden credential or transient file: {relative.as_posix()}"
            )
        files.append(path)
    if not files:
        raise ValueError(f"result artifact root contains no files: {root}")
    return tuple(files)


def _source_specs(
    *,
    release_root: Path,
    output_root: Path,
    roots_by_kind: dict[ResultArtifactKind, Sequence[Path]],
) -> tuple[tuple[ResultExportSourceV1, Path, tuple[Path, ...]], ...]:
    specs: list[tuple[ResultExportSourceV1, Path, tuple[Path, ...]]] = []
    observed: set[tuple[ResultArtifactKind, str]] = set()
    for kind in _RESULT_ARTIFACT_KINDS:
        roots = roots_by_kind[kind]
        for root_path in roots:
            if root_path.is_symlink():
                raise ValueError(f"result artifact root must not be a symbolic link: {root_path}")
            root = root_path.resolve()
            if root.is_relative_to(release_root):
                raise ValueError("result artifacts must remain outside the immutable release")
            if output_root.is_relative_to(root) or root.is_relative_to(output_root):
                raise ValueError("result artifacts and output directory must not contain one another")
            label = _safe_label(root)
            identity = (kind, label)
            if identity in observed:
                raise ValueError(f"duplicate result artifact label for {kind}: {label}")
            observed.add(identity)
            files = _artifact_files(root)
            prefix = f"artifacts/{kind}/{label}"
            specs.append(
                (
                    ResultExportSourceV1(
                        kind=kind,
                        label=label,
                        archive_prefix=prefix,
                        file_count=len(files),
                    ),
                    root,
                    files,
                )
            )
    return tuple(specs)


def _members(
    specs: Iterable[tuple[ResultExportSourceV1, Path, tuple[Path, ...]]],
) -> tuple[ResultExportMemberV1, ...]:
    members: list[ResultExportMemberV1] = []
    for source, root, files in specs:
        for path in files:
            archive_path = f"{source.archive_prefix}/{path.relative_to(root).as_posix()}"
            members.append(
                ResultExportMemberV1(
                    path=archive_path,
                    sha256=_sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
    return tuple(sorted(members, key=lambda member: member.path))


def _json_bytes(model: BaseModel) -> bytes:
    return (json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _zip_info(path: str) -> ZipInfo:
    info = ZipInfo(path, date_time=_FIXED_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_member(archive: ZipFile, *, member: str, source: Path) -> None:
    with archive.open(_zip_info(member), "w", force_zip64=True) as destination, source.open("rb") as stream:
        shutil.copyfileobj(stream, destination, length=1024 * 1024)


def _write_archive(
    *,
    path: Path,
    manifest: ResultExportBundleManifestV1,
    specs: tuple[tuple[ResultExportSourceV1, Path, tuple[Path, ...]], ...],
) -> None:
    manifest_bytes = _json_bytes(manifest)
    source_by_prefix = {source.archive_prefix: (root, files) for source, root, files in specs}
    checksum_rows = [f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json"]
    checksum_rows.extend(f"{member.sha256}  {member.path}" for member in manifest.members)
    checksums = ("\n".join(checksum_rows) + "\n").encode("utf-8")
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        archive.writestr(_zip_info("manifest.json"), manifest_bytes)
        archive.writestr(_zip_info("SHA256SUMS"), checksums)
        for source in manifest.sources:
            root, files = source_by_prefix[source.archive_prefix]
            for source_file in files:
                member = f"{source.archive_prefix}/{source_file.relative_to(root).as_posix()}"
                _write_member(archive, member=member, source=source_file)


def export_results_v1(
    *,
    release_root: Path,
    output_dir: Path,
    run_roots: Sequence[Path],
    grade_roots: Sequence[Path],
    verification_roots: Sequence[Path],
    analysis_roots: Sequence[Path] = (),
) -> Path:
    """Build one deterministic result archive outside the immutable release."""

    supplied_release = Path(release_root)
    if supplied_release.is_symlink():
        raise ValueError("release root must not be a symbolic link")
    release = supplied_release.resolve()
    output = Path(output_dir).resolve()
    if not release.is_dir() or release.is_symlink():
        raise ValueError("release root must be a regular directory")
    if output.exists():
        raise FileExistsError(f"result export output already exists: {output}")
    if output.is_relative_to(release):
        raise ValueError("result export output must remain outside the immutable release")
    release_manifest, export_contract_sha256 = _release_contract(release)
    specs = _source_specs(
        release_root=release,
        output_root=output,
        roots_by_kind={
            "run": run_roots,
            "grade": grade_roots,
            "verification": verification_roots,
            "analysis": analysis_roots,
        },
    )
    members = _members(specs)
    manifest = ResultExportBundleManifestV1(
        release_id=release_manifest.release_id,
        release_stage=release_manifest.release_stage,
        release_manifest_sha256=_sha256_file(release / "RELEASE_MANIFEST.json"),
        export_contract_sha256=export_contract_sha256,
        sources=tuple(source for source, _, _ in specs),
        members=members,
        member_count=len(members),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        archive_name = f"{release_manifest.release_id}-results.zip"
        archive_path = staged / archive_name
        _write_archive(path=archive_path, manifest=manifest, specs=specs)
        archive_sha256 = _sha256_file(archive_path)
        (staged / f"{archive_name}.sha256").write_text(
            f"{archive_sha256}  {archive_name}\n",
            encoding="utf-8",
        )
        (staged / "result_export_receipt.json").write_bytes(_json_bytes(manifest))
        os.replace(staged, output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public result exporter."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = export_results_v1(
        release_root=args.release_root,
        output_dir=args.output_dir,
        run_roots=args.run_root,
        grade_roots=args.grade_root,
        verification_roots=args.verification_root,
        analysis_roots=args.analysis_root,
    )
    print(output)
    return 0


__all__ = ["export_results_v1", "main"]
