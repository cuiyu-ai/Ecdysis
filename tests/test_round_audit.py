import unittest
from pathlib import Path

from ecdysis.pipeline.steps import append_round_record


class RoundAuditTest(unittest.TestCase):
    def test_records_requested_calls_and_candidate_status(self) -> None:
        records = []
        append_round_record(
            records,
            round_num=1,
            train_dir=Path("train"),
            failures=[{"task_id": "1"}],
            analysis={
                "skills": [{"id": "skill"}],
                "code_changes": [{"target": "modify"}],
                "evolution_audit": {
                    "mode": "batched_mad",
                    "opencode_calls_requested": 1,
                    "mad_calls_requested": 7,
                },
            },
        )

        record = records[0]
        self.assertEqual(record["evolution_audit"]["opencode_calls_requested"], 1)
        self.assertEqual(record["evolution_audit"]["mad_calls_requested"], 7)
        self.assertEqual(record["patch_decision"]["status"], "candidate")

