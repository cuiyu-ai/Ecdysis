import json

import pytest

from ecdysis.fdcr import review


def _failure(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "trial": 0,
        "reward": 0.0,
        "termination_reason": "stopped",
        "reward_breakdown": {"execution": 0.0},
        "messages": [{"role": "assistant", "content": "failed action"}],
    }


class _FakeClient:
    model = "fake/fdcr"

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def usage_event_count(self):
        return self.calls

    def usage_summary(self, start_index=0):
        return {"calls": self.calls - start_index}

    def chat(self, _messages, temperature=0.3):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("controlled FDCR interruption")
        return f"reply-{self.calls}"

    def chat_json(self, _messages, temperature=0.0):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("controlled moderator interruption")
        return {"proposed_changes": [], "skills": []}


def test_fdcr_checkpoint_resumes_at_next_turn(tmp_path):
    checkpoint = tmp_path / "fdcr.json"
    failures = [_failure("alpha")]
    groups = {"stopped:execution": failures}

    first = _FakeClient(fail_at=3)
    with pytest.raises(RuntimeError, match="FDCR interruption"):
        review.run_harness_fdcr(
            failures,
            groups,
            client=first,
            scope="example",
            checkpoint_path=checkpoint,
        )

    state = json.loads(checkpoint.read_text())
    assert state["status"] == "interrupted"
    assert state["next_turn_index"] == 2
    assert len(state["transcript"]) == 2

    resumed_client = _FakeClient()
    result = review.run_harness_fdcr(
        failures,
        groups,
        client=resumed_client,
        scope="example",
        checkpoint_path=checkpoint,
    )
    assert resumed_client.calls == 5
    assert result["checkpoint_status"] == "completed"
    assert len(result["transcript"]) == 7

    completed_client = _FakeClient(fail_at=1)
    reused = review.run_harness_fdcr(
        failures,
        groups,
        client=completed_client,
        scope="example",
        checkpoint_path=checkpoint,
    )
    assert completed_client.calls == 0
    assert reused["checkpoint_status"] == "completed"
