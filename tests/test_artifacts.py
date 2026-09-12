from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
            experiment="mixed-training",
            scope="primary",
            round=2,
            mode="mixed_training",
            skills=[skill("structured-errors")],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_skill_artifact(artifact, Path(directory) / "artifact.json")
            loaded = load_skill_artifact(path)

        self.assertEqual(loaded.to_dict(), artifact.to_dict())

    def test_merge_deduplicates_skills_and_rejects_wrong_domain(self) -> None:
        first = SkillArtifact(
            "mixed-training", "primary", 1, "mixed_training", [skill("a")]
        )
        second = SkillArtifact(
            "mixed-training",
            "primary",
            2,
            "mixed_training",
            [skill("a"), skill("b")],
        )

        merged = merge_skill_artifacts(
            experiment="mixed-training",
            scope="primary",
            round_num=3,
            mode="mixed_training",
            artifacts=[first, second],
        )

        self.assertEqual([item.id for item in merged.skills], ["a", "b"])
        with self.assertRaises(ValueError):
            first.validate_for_scope("secondary")


if __name__ == "__main__":
    unittest.main()
