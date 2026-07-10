from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecdysis.evolution import (
    collect_failed_trajectories,
    format_failure_groups,
    group_failures_by_pattern,
)


class CrossInstanceEvolutionTest(unittest.TestCase):
    def test_collects_only_failed_simulations_with_compact_messages(self) -> None:
        results = {
            "tasks": [{"id": "task-1", "instruction": "book a flight"}],
            "simulations": [
                {
                    "task_id": "task-1",
                    "trial": 1,
                    "reward_info": {
                        "reward": 0.0,
                        "reward_breakdown": {"database": 0.0},
                    },
                    "termination_reason": "tool_error",
                    "messages": [{"role": "assistant", "content": "x" * 3000}],
                },
                {
                    "task_id": "task-1",
                    "trial": 2,
                    "reward_info": {"reward": 1.0},
                    "messages": [],
                },
            ],
        }

        failures = collect_failed_trajectories(results, max_messages=1)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["task"], results["tasks"][0])
        self.assertLessEqual(len(failures[0]["messages"][0]["content"]), 2000)

    def test_groups_failures_by_termination_and_failed_reward_basis(self) -> None:
        failures = [
            {
                "task_id": "1",
                "termination_reason": "tool_error",
                "reward_breakdown": {"database": 0.0, "communication": 1.0},
            },
            {
                "task_id": "2",
                "termination_reason": "tool_error",
                "reward_breakdown": {"database": 0.0, "communication": 1.0},
            },
            {
                "task_id": "3",
                "termination_reason": "timeout",
                "reward_breakdown": {},
            },
        ]

        groups = group_failures_by_pattern(failures)
        summary = format_failure_groups(groups)

        self.assertEqual(len(groups["tool_error:database"]), 2)
        self.assertIn("2 failures across 2 distinct tasks", summary)
        self.assertIn("timeout:no_breakdown", summary)


if __name__ == "__main__":
    unittest.main()
