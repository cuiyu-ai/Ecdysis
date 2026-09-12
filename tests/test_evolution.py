from __future__ import annotations

import unittest

from ecdysis.evolution import (
    collect_failed_trajectories,
    format_failure_groups,
    group_failures_by_pattern,
)


class CrossInstanceEvolutionTest(unittest.TestCase):
    def test_collects_only_failed_simulations_with_compact_messages(self) -> None:
        results = {
            "tasks": [{"id": "task-1", "instruction": "complete the request"}],
            "simulations": [
                {
                    "task_id": "task-1",
                    "trial": 1,
                    "reward_info": {
                        "reward": 0.0,
                        "reward_breakdown": {"execution": 0.0},
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
                "reward_breakdown": {"execution": 0.0, "communication": 1.0},
            },
            {
                "task_id": "2",
                "termination_reason": "tool_error",
                "reward_breakdown": {"execution": 0.0, "communication": 1.0},
            },
            {
                "task_id": "3",
                "termination_reason": "timeout",
                "reward_breakdown": {},
            },
        ]

        groups = group_failures_by_pattern(failures)
        summary = format_failure_groups(groups)

        self.assertEqual(len(groups["tool_error:execution"]), 2)
        self.assertIn("2 failures across 2 distinct tasks", summary)
        self.assertIn("timeout:no_breakdown", summary)

    def test_zero_message_limit_removes_message_content(self) -> None:
        results = {
            "simulations": [
                {
                    "task_id": "task-1",
                    "reward_info": {"reward": 0.0},
                    "messages": [{"role": "assistant", "content": "evidence"}],
                }
            ]
        }

        failures = collect_failed_trajectories(results, max_messages=0)

        self.assertEqual(failures[0]["messages"], [])

    def test_failure_threshold_uses_strict_less_than(self) -> None:
        results = {
            "simulations": [
                {
                    "task_id": name,
                    "reward_info": {"reward": reward},
                    "messages": [],
                }
                for name, reward in (("below", 0.49), ("equal", 0.5), ("above", 0.9))
            ]
        }

        failures = collect_failed_trajectories(
            results,
            failure_threshold=0.5,
        )

        self.assertEqual([failure["task_id"] for failure in failures], ["below"])

    def test_recurring_groups_are_prioritized_over_singletons(self) -> None:
        failures = [
            {
                "task_id": "one",
                "termination_reason": "isolated",
                "reward_breakdown": {},
            },
            {
                "task_id": "two",
                "termination_reason": "recurring",
                "reward_breakdown": {},
            },
            {
                "task_id": "three",
                "termination_reason": "recurring",
                "reward_breakdown": {},
            },
        ]

        groups = group_failures_by_pattern(failures)

        self.assertEqual(next(iter(groups)), "recurring:no_breakdown")


if __name__ == "__main__":
    unittest.main()
