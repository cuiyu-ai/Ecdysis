"""Synthetic end-to-end example for Ecdysis training and inference."""

from __future__ import annotations

from ecdysis.training import EcdysisTrainer, TrainingConfig


class ReviewClient:
    model = "synthetic-reviewer"

    def chat(self, _messages, temperature=0.3):
        return "Prefer the smallest update supported by recurring failures."

    def chat_json(self, _messages, temperature=0.0):
        return {
            "failure_patterns": ["repeated execution failure"],
            "proposed_changes": [
                {
                    "component": "runtime",
                    "description": "apply the validated correction",
                    "rationale": "the pattern recurs across tasks",
                }
            ],
            "skills": [],
        }


def collect(harness):
    reward = float(harness["revision"] >= 1)
    return {
        "tasks": [{"id": "one"}, {"id": "two"}],
        "simulations": [
            {
                "task_id": task_id,
                "reward_info": {
                    "reward": reward,
                    "reward_breakdown": {"execution": reward},
                },
                "termination_reason": None if reward else "stopped",
                "messages": [],
            }
            for task_id in ("one", "two")
        ],
    }


def edit(harness, _specification, _failures, _groups):
    return {"revision": harness["revision"] + 1}


trainer = EcdysisTrainer(
    collector=collect,
    editor=edit,
    review_client=ReviewClient(),
    config=TrainingConfig(rounds=2, refinement_passes=2),
)
result = trainer.train({"revision": 0})
inference = result.infer_many(
    [{"id": "example task"}, {"id": "second task"}],
    lambda harness, task: f"{task['id']} @ revision {harness['revision']}",
)

print(f"final_score={result.final_score:.1f}")
for record in inference.records:
    print(record.output)
