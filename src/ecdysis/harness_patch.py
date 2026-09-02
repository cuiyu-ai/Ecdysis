"""H2/H3/H4 harness code patches produced by evolution.

Evolution experiments edit ``tau2/harness/<domain>.py`` directly.  Each
round's edit set is captured as a ``HarnessPatch`` JSON document and
written under ``data/evolved_harness/<exp>/<domain>/<timestamp>/``.

The orchestrator applies a patch by writing the new file content for
each entry. Before evolution, it copies the complete harness tree to a
temporary directory so a regression can be reverted without touching Git
or a user's uncommitted changes.
"""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.getenv("ECDYSIS_RUNTIME_ROOT", str(PACKAGE_ROOT))
).expanduser()
HARNESS_DIR = PROJECT_ROOT / "tau2" / "harness"


@dataclass
class HarnessSnapshot:
    """Temporary, complete copy of a harness tree for one experiment run."""

    snapshot_dir: Path
    _temporary_dir: tempfile.TemporaryDirectory = field(repr=False)

    @classmethod
    def capture(cls, harness_dir: Path) -> "HarnessSnapshot":
        if not harness_dir.is_dir():
            raise FileNotFoundError(f"Harness directory not found: {harness_dir}")
        temporary_dir = tempfile.TemporaryDirectory(prefix="ecdysis-harness-")
        snapshot_dir = Path(temporary_dir.name) / "harness"
        shutil.copytree(harness_dir, snapshot_dir, copy_function=shutil.copy2)
        return cls(snapshot_dir=snapshot_dir, _temporary_dir=temporary_dir)

    def restore(self, harness_dir: Path) -> None:
        """Restore the exact tree captured by :meth:`capture`.

        Files created by an evolution patch are removed, while every original
        file -- including user edits that predated the experiment -- is copied
        back from the temporary snapshot.
        """
        if not self.snapshot_dir.is_dir():
            raise RuntimeError("Harness snapshot is no longer available")

        harness_dir.mkdir(parents=True, exist_ok=True)
        snapshot_entries = {
            path.relative_to(self.snapshot_dir)
            for path in self.snapshot_dir.rglob("*")
        }
        current_entries = sorted(
            harness_dir.rglob("*"), key=lambda path: len(path.parts), reverse=True
        )
        for path in current_entries:
            if path.relative_to(harness_dir) in snapshot_entries:
                continue
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

        shutil.copytree(
            self.snapshot_dir,
            harness_dir,
            dirs_exist_ok=True,
            copy_function=shutil.copy2,
        )

    def cleanup(self) -> None:
        self._temporary_dir.cleanup()


def snapshot_harness_files(
    harness_dir: Path | None = None,
) -> dict[str, str]:
    """Return a copy of all ``*.py`` contents under ``harness_dir``."""
    root = harness_dir or HARNESS_DIR
    return {path.name: path.read_text() for path in sorted(root.glob("*.py"))}


def restore_harness_files(
    snapshot: dict[str, str],
    *,
    harness_dir: Path | None = None,
) -> None:
    """Restore the exact top-level Python-file snapshot."""
    root = harness_dir or HARNESS_DIR
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("*.py"):
        if path.name not in snapshot:
            path.unlink()
    for name, content in snapshot.items():
        _resolve_harness_target(root, name).write_text(content)


def copy_harness_py_files(source_dir: Path, dest_dir: Path) -> None:
    """Copy ``*.py`` harness modules from ``source_dir`` into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob("*.py"):
        dest_dir.joinpath(path.name).write_text(path.read_text())


@dataclass
class HarnessFileChange:
    """A single file edit produced by evolution."""

    file_path: str
    """Path relative to ``tau2/harness/`` (e.g. ``"airline.py"``)."""

    new_content: str
    """Full new file content.  Evolution rewrites the file in full,
    not a unified diff, so the apply step is a single write."""

    description: str = ""
    """Human-readable note from the evolution LLM explaining the change."""

    target: str = ""
    """What the change targets (``"h2_rule"``, ``"h3_hint"``,
    ``"h4_annotator"``, ``"import"`` ...).  Free-form string."""


@dataclass
class HarnessPatch:
    """A versioned edit set for the H2/H3/H4 harness source files."""

    experiment: str
    domain: str
    round: int
    file_changes: list[HarnessFileChange] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_patch: str | None = None
    """Path of the previous round's HarnessPatch JSON, for lineage."""

    skill_artifact: str | None = None
    """Path of the SkillArtifact JSON for this round (H5 skills).
    Stored alongside the patch so the round is fully self-describing."""

    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessPatch":
        for key in ("experiment", "domain", "round"):
            if key not in data:
                raise ValueError(f"HarnessPatch missing required field: {key}")
        changes = [
            HarnessFileChange(
                file_path=str(change["file_path"]),
                new_content=str(change["new_content"]),
                description=str(change.get("description", "")),
                target=str(change.get("target", "")),
            )
            for change in data.get("file_changes", [])
        ]
        return cls(
            experiment=str(data["experiment"]),
            domain=str(data["domain"]),
            round=int(data["round"]),
            file_changes=changes,
            created_at=str(data.get("created_at") or datetime.now().isoformat()),
            parent_patch=data.get("parent_patch"),
            skill_artifact=data.get("skill_artifact"),
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )

    def resolve_targets(self, harness_dir: Path = HARNESS_DIR) -> list[Path]:
        return [
            _resolve_harness_target(harness_dir, change.file_path)
            for change in self.file_changes
        ]


def save_patch(patch: HarnessPatch, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(patch.to_dict(), indent=2))
    return out


def load_patch(path: str | Path) -> HarnessPatch:
    return HarnessPatch.from_dict(json.loads(Path(path).read_text()))


def _resolve_harness_target(harness_dir: Path, file_path: str) -> Path:
    """Resolve one top-level Python module without allowing path traversal."""
    raw = str(file_path).strip()
    posix_path = PurePosixPath(raw.replace("\\", "/"))
    windows_path = PureWindowsPath(raw)
    if (
        not raw
        or "\x00" in raw
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or len(posix_path.parts) != 1
        or posix_path.name in {".", ".."}
        or posix_path.suffix.lower() != ".py"
    ):
        raise ValueError(
            "Harness patch paths must name one top-level Python file: "
            f"{file_path!r}"
        )

    root = harness_dir.resolve()
    target = (root / posix_path.name).resolve()
    if target.parent != root:
        raise ValueError(f"Harness patch path escapes the harness directory: {file_path!r}")
    return target


def _check_python_content(content: str, path: Path) -> None:
    """Raise before writing when candidate Python source is invalid.

    Catches syntax errors / indentation errors / bad encoding in any
    harness file the coding agent just wrote, before the next eval
    subprocess blows up on import.
    """
    try:
        compile(content, str(path), "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"harness file {path.name} has invalid Python syntax "
            f"(rejected before write). Patch was not applied.\n"
            f"{exc}"
        ) from exc


def apply_patch(patch: HarnessPatch, *, project_root: Path = PROJECT_ROOT) -> None:
    """Write the new content for each file in ``patch.file_changes``.

    Every target and source file is validated before the first write. If any
    write fails, files already written by this call are restored immediately.
    """
    harness_dir = project_root / "tau2" / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[HarnessFileChange, Path]] = []
    seen_targets: set[Path] = set()
    for change in patch.file_changes:
        target = _resolve_harness_target(harness_dir, change.file_path)
        if target in seen_targets:
            raise ValueError(f"Duplicate harness patch target: {change.file_path!r}")
        seen_targets.add(target)
        _check_python_content(change.new_content, target)
        prepared.append((change, target))

    originals = {
        target: target.read_bytes() if target.exists() else None
        for _, target in prepared
    }
    pending_paths: list[Path] = []
    try:
        for change, target in prepared:
            pending = target.with_name(f".{target.name}.ecdysis.tmp")
            pending_paths.append(pending)
            pending.write_text(change.new_content)
            pending.replace(target)
            print(
                f"  [harness patch] wrote "
                f"{target.relative_to(project_root)} ({change.target})"
            )
    except Exception:
        for target, original in originals.items():
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original)
        raise
    finally:
        for pending in pending_paths:
            pending.unlink(missing_ok=True)


def revert_harness(
    snapshot: HarnessSnapshot,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Restore ``tau2/harness`` from an experiment-local temporary snapshot."""
    snapshot.restore(project_root / "tau2" / "harness")


def snapshot_harness(*, project_root: Path = PROJECT_ROOT) -> HarnessSnapshot:
    """Capture ``tau2/harness`` without staging, resetting, or reading Git."""
    return HarnessSnapshot.capture(project_root / "tau2" / "harness")
