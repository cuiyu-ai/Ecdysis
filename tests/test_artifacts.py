from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecdysis.artifacts import (
    EvolvedSkill,
    SkillArtifact,
    load_skill_artifact,
    merge_skill_artifacts,
    save_skill_artifact,
)


def skill(skill_id: str) -> EvolvedSkill:
    return EvolvedSkill(
        id=skill_id,
        title=f"Skill {skill_id}",
        pattern="Return structured tool errors.",
        tip="Preserve the original query parameters.",
    )


class ArtifactTest(unittest.TestCase):
    def test_skill_artifact_round_trips_through_json(self) -> None:
        artifact = SkillArtifact(
            experiment="E4",
            domain="airline",
            round=2,
            mode="cross_instance",
            skills=[skill("structured-errors")],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_skill_artifact(artifact, Path(directory) / "artifact.json")
            loaded = load_skill_artifact(path)

        self.assertEqual(loaded.to_dict(), artifact.to_dict())

    def test_merge_deduplicates_skills_and_rejects_wrong_domain(self) -> None:
        first = SkillArtifact("E4", "airline", 1, "cross_instance", [skill("a")])
        second = SkillArtifact(
            "E4", "airline", 2, "cross_instance", [skill("a"), skill("b")]
        )

        merged = merge_skill_artifacts(
            experiment="E4",
            domain="airline",
            round_num=3,
            mode="cross_instance",
            artifacts=[first, second],
        )

        self.assertEqual([item.id for item in merged.skills], ["a", "b"])
        with self.assertRaises(ValueError):
            first.validate_for_domain("retail")


if __name__ == "__main__":
    unittest.main()
