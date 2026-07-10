"""H2/H3/H4 harness code patches produced by evolution.

Evolution experiments edit ``tau2/harness/<domain>.py`` directly.  Each
round's edit set is captured as a ``HarnessPatch`` JSON document and
written under ``data/evolved_harness/<exp>/<domain>/<timestamp>/``.

The orchestrator applies a patch by writing the new file content for
each entry.  A git snapshot of the harness is taken before every round
so any regression can be reverted with a single ``git checkout``.
"""

from __future__ import annotations

import json
import py_compile
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = PROJECT_ROOT / "tau2" / "harness"


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
    """Restore harness ``*.py`` files from ``snapshot_harness_files`` output."""
    root = harness_dir or HARNESS_DIR
    for name, content in snapshot.items():
        (root / name).write_text(content)


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

    def resolve_targets(self) -> list[Path]:
        return [HARNESS_DIR / change.file_path for change in self.file_changes]


def save_patch(patch: HarnessPatch, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(patch.to_dict(), indent=2))
    return out


def load_patch(path: str | Path) -> HarnessPatch:
    return HarnessPatch.from_dict(json.loads(Path(path).read_text()))


def _check_python_syntax(path: Path) -> None:
    """Raise ``py_compile.PyCompileError`` if ``path`` is not valid Python.

    Catches syntax errors / indentation errors / bad encoding in any
    harness file the coding agent just wrote, before the next eval
    subprocess blows up on import.
    """
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        raise RuntimeError(
            f"harness file {path.name} has invalid Python syntax "
            f"(rejected by py_compile). Patch will be reverted.\n"
            f"{exc}"
        ) from exc


def apply_patch(patch: HarnessPatch, *, project_root: Path = PROJECT_ROOT) -> None:
    """Write the new content for each file in ``patch.file_changes``.

    Existing files are overwritten in place.  Callers MUST snapshot the
    harness first (see ``snapshot_harness``) so this is reversible.
    """
    harness_dir = project_root / "tau2" / "harness"
    for change in patch.file_changes:
        target = harness_dir / change.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(change.new_content)
        if target.suffix == ".py":
            _check_python_syntax(target)
        print(f"  [harness patch] wrote {target.relative_to(project_root)} ({change.target})")


def revert_harness(*, project_root: Path = PROJECT_ROOT) -> None:
    """Restore the tau2/harness tree from the latest git snapshot.

    Uses ``git checkout HEAD -- tau2/harness`` to discard any uncommitted
    edits evolution may have made.  This is the only safe revert path:
    rewriting file content from in-memory state is fragile.
    """
    cmd = ["git", "checkout", "HEAD", "--", "tau2/harness"]
    result = subprocess.run(
        cmd, cwd=project_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to revert tau2/harness: {result.stderr or result.stdout}"
        )


def snapshot_harness(*, project_root: Path = PROJECT_ROOT) -> None:
    """Stage the current tau2/harness state so ``revert_harness`` works.

    Without this, edits in the working tree can be lost on a non-clean
    checkout.  We use ``git add`` to make the working-tree content match
    HEAD, then ``git checkout HEAD --`` can rewind it.  This is a no-op
    when the harness is already clean.
    """
    add = subprocess.run(
        ["git", "add", "--", "tau2/harness"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise RuntimeError(
            f"Failed to stage tau2/harness snapshot: {add.stderr or add.stdout}"
        )
