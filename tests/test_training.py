from ecdysis.training import EcdysisTrainer, TrainingConfig


class _ReviewClient:
    model = "test-reviewer"

    def __init__(self):
        self.calls = 0

    def chat(self, _messages, temperature=0.3):
        self.calls += 1
        return "review"

    def chat_json(self, _messages, temperature=0.0):
        self.calls += 1
        return {"proposed_changes": [{"component": "runtime"}], "skills": []}


def _results(score):
    return {
        "simulations": [
            {
                "task_id": task_id,
                "reward_info": {
                    "reward": score,
                    "reward_breakdown": {"execution": score},
                },
                "termination_reason": None if score == 1.0 else "stopped",
                "messages": [],
            }
            for task_id in ("one", "two")
        ]
    }


def test_training_accepts_only_an_improving_candidate():
    client = _ReviewClient()
    trainer = EcdysisTrainer(
        collector=lambda harness: _results(float(harness > 0)),
        editor=lambda harness, _spec, _failures, _groups: harness + 1,
        review_client=client,
        config=TrainingConfig(rounds=2),
    )

    result = trainer.train(0)

    assert result.frozen_harness == 1
    assert result.final_score == 1.0
    assert result.rounds[0].status == "accepted"
    assert result.rounds[0].baseline_score == 0.0
    assert result.rounds[0].retained_score == 1.0
    assert result.rounds[1].status == "no_failures"
    assert result.infer("task", lambda harness, task: (harness, task)) == (1, "task")
    assert client.calls == 7


def test_training_rejects_a_non_improving_candidate():
    trainer = EcdysisTrainer(
        collector=lambda harness: _results(0.5 if harness == 0 else 0.0),
        editor=lambda _harness, _spec, _failures, _groups: -1,
        review_client=_ReviewClient(),
        config=TrainingConfig(rounds=1, failure_threshold=1.0),
    )

    result = trainer.train(0)

    assert result.frozen_harness == 0
    assert result.final_score == 0.5
    assert result.rounds[0].status == "rejected"
    assert result.rounds[0].baseline_score == 0.5
    assert result.rounds[0].retained_score == 0.5
