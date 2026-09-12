"""Structured artifacts produced by Ecdysis evolution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

ArtifactMode = Literal["serial", "mixed_training", "fdcr", "manual"]


@dataclass
class EvolvedSkill:
    """A reusable behavior learned from failure evidence."""

    id: str
    title: str
    pattern: str
    tip: str
    source: str = "evolution"
    source_task_ids: list[str] = field(default_factory=list)
    issue_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolvedSkill:
        required = ("id", "title", "pattern", "tip")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Evolved skill missing required fields: {missing}")
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            pattern=str(data["pattern"]),
            tip=str(data["tip"]),
            source=str(data.get("source") or "evolution"),
            source_task_ids=[str(item) for item in data.get("source_task_ids", [])],
            issue_type=data.get("issue_type"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SkillArtifact:
    """Versioned collection of evolved skills for one task scope."""

    experiment: str
    scope: str
    round: int
    mode: ArtifactMode
    skills: list[EvolvedSkill] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillArtifact:
        required = ("experiment", "scope", "round", "mode")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Skill artifact missing required fields: {missing}")
        return cls(
            experiment=str(data["experiment"]),
            scope=str(data["scope"]),
            round=int(data["round"]),
            mode=data["mode"],
            skills=[EvolvedSkill.from_dict(item) for item in data.get("skills", [])],
            created_at=str(data.get("created_at") or datetime.now().isoformat()),
            parent_artifacts=[str(item) for item in data.get("parent_artifacts", [])],
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_for_scope(self, scope: str) -> None:
        if self.scope != scope:
            raise ValueError(
                f"Artifact scope mismatch: expected {scope}, got {self.scope}"
            )


def load_skill_artifact(path: str | Path) -> SkillArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("skill artifact must contain a JSON object")
    return SkillArtifact.from_dict(data)


def save_skill_artifact(artifact: SkillArtifact, path: str | Path) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), indent=2), encoding="utf-8"
    )
    return artifact_path


def load_evolved_skills(
    paths: list[str | Path], scope: str | None = None
) -> list[EvolvedSkill]:
    """Load and deduplicate skills from one or more artifacts."""
    skills: list[EvolvedSkill] = []
    seen_ids: set[str] = set()
    for path in paths:
        artifact = load_skill_artifact(path)
        if scope is not None:
            artifact.validate_for_scope(scope)
        for skill in artifact.skills:
            if skill.id not in seen_ids:
                seen_ids.add(skill.id)
                skills.append(skill)
    return skills


def merge_skill_artifacts(
    *,
    experiment: str,
    scope: str,
    round_num: int,
    mode: ArtifactMode,
    artifacts: list[SkillArtifact],
    new_skills: list[EvolvedSkill] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SkillArtifact:
    """Merge parent artifacts and new skills, deduplicating by identifier."""
    merged: list[EvolvedSkill] = []
    seen_ids: set[str] = set()
    parents: list[str] = []
    for artifact in artifacts:
        artifact.validate_for_scope(scope)
        parents.extend(artifact.parent_artifacts)
        for skill in artifact.skills:
            if skill.id not in seen_ids:
                seen_ids.add(skill.id)
                merged.append(skill)
    for skill in new_skills or []:
        if skill.id not in seen_ids:
            seen_ids.add(skill.id)
            merged.append(skill)

    return SkillArtifact(
        experiment=experiment,
        scope=scope,
        round=round_num,
        mode=mode,
        skills=merged,
        parent_artifacts=parents,
        metadata=metadata or {},
    )
