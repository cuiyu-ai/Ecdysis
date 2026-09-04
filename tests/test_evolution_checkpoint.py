import json

import pytest

from ecdysis import evolution
from ecdysis.mad import debate


def _failure(task_id: str, marker: str) -> dict:
    return {
        "task_id": task_id,
        "trial": 0,
        "reward": 0.0,
        "termination_reason": "user_stop",
        "reward_breakdown": {"DB": 0.0},
        "messages": [{"role": "assistant", "content": marker}],
    }


def test_serial_checkpoint_isolates_failure_and_resumes(
    tmp_path, monkeypatch
):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "retail.py").write_text("BASE = True\n")
    (harness / "base.py").write_text("SHARED = True\n")
    checkpoint = tmp_path / "checkpoint"
    failures = [
        _failure("alpha", "ONLY-TASK-ALPHA"),
        _failure("beta", "ONLY-TASK-BETA"),
    ]

    monkeypatch.setattr(evolution, "_read_harness_dir", lambda: harness)
    monkeypatch.setattr(
        evolution,
        "_filter_replay_safe_changes",
        lambda changes, **_: changes,
    )

    calls = []

    def first_pass(prompt, work_dir, **kwargs):
        calls.append(prompt)
        marker = "ALPHA" if "ONLY-TASK-ALPHA" in prompt else "BETA-PARTIAL"
        target = work_dir / "retail.py"
        target.write_text(target.read_text() + f"{marker} = True\n")
        audit = kwargs["audit_events"]
        if marker == "BETA-PARTIAL":
            audit.append({"success": False, "error_type": "timeout", "attempt": 1})
            raise TimeoutError("controlled timeout")
        audit.append({"success": True, "attempt": 1, "duration_seconds": 1.0})
        return "{}"

    monkeypatch.setattr(evolution, "_call_opencode", first_pass)
    with pytest.raises(RuntimeError, match="round is incomplete"):
        evolution.analyze_failures_serial(
            failures,
            domain="retail",
            checkpoint_dir=checkpoint,
            timeout=600,
            max_retries=1,
        )

    workspace_text = (checkpoint / "workspace" / "retail.py").read_text()
    assert "ALPHA = True" in workspace_text
    assert "BETA-PARTIAL" not in workspace_text
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert len(manifest["completed"]) == 1
    assert len(manifest["failed_attempts"]) == 1
    assert list((checkpoint / "failed_attempts").glob("*/retail.py"))

    resumed_prompts = []

    def resume_pass(prompt, work_dir, **kwargs):
        resumed_prompts.append(prompt)
        target = work_dir / "retail.py"
        target.write_text(target.read_text() + "BETA = True\n")
        kwargs["audit_events"].append(
            {"success": True, "attempt": 1, "duration_seconds": 1.0}
        )
        return "{}"

    monkeypatch.setattr(evolution, "_call_opencode", resume_pass)
    resumed = evolution.analyze_failures_serial(
        failures,
        domain="retail",
        checkpoint_dir=checkpoint,
        timeout=600,
        max_retries=1,
    )

    assert len(resumed_prompts) == 1
    assert "ONLY-TASK-BETA" in resumed_prompts[0]
    assert "ONLY-TASK-ALPHA" not in resumed_prompts[0]
    assert len(resumed["completed_failure_keys"]) == 2
    final_text = (checkpoint / "workspace" / "retail.py").read_text()
    assert "ALPHA = True" in final_text
    assert "BETA = True" in final_text


def test_serial_checkpoint_rejects_changed_baseline(tmp_path, monkeypatch):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "retail.py").write_text("BASE = 1\n")
    checkpoint = tmp_path / "checkpoint"
    failure = _failure("alpha", "ONLY-TASK-ALPHA")

    monkeypatch.setattr(evolution, "_read_harness_dir", lambda: harness)
    monkeypatch.setattr(
        evolution,
        "_filter_replay_safe_changes",
        lambda changes, **_: changes,
    )

    def success(_prompt, work_dir, **kwargs):
        kwargs["audit_events"].append(
            {"success": True, "attempt": 1, "duration_seconds": 1.0}
        )
        return "{}"

    monkeypatch.setattr(evolution, "_call_opencode", success)
    evolution.analyze_failures_serial(
        [failure], domain="retail", checkpoint_dir=checkpoint
    )
    (harness / "retail.py").write_text("BASE = 2\n")

    with pytest.raises(RuntimeError, match="does not match"):
        evolution.analyze_failures_serial(
            [failure], domain="retail", checkpoint_dir=checkpoint
        )


def test_batch_checkpoint_resumes_only_interrupted_call(tmp_path, monkeypatch):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "retail.py").write_text("BASE = True\n")
    checkpoint = tmp_path / "batch"
    monkeypatch.setattr(
        evolution,
        "_filter_replay_safe_changes",
        lambda changes, **_: changes,
    )

    def interrupted(_prompt, work_dir, **kwargs):
        target = work_dir / "retail.py"
        target.write_text(target.read_text() + "PARTIAL = True\n")
        kwargs["audit_events"].append(
            {"success": False, "error_type": "timeout", "attempt": 1}
        )
        raise TimeoutError("controlled")

    monkeypatch.setattr(evolution, "_call_opencode", interrupted)
    with pytest.raises(TimeoutError):
        evolution._run_opencode_analysis(
            prompt="batch-prompt",
            source_harness_dir=harness,
            model="provider/model",
            domain="retail",
            checkpoint_dir=checkpoint,
        )
    assert "PARTIAL" not in (checkpoint / "workspace" / "retail.py").read_text()
    state = json.loads((checkpoint / "manifest.json").read_text())
    assert state["status"] == "interrupted"

    calls = []

    def completed(_prompt, work_dir, **kwargs):
        calls.append(1)
        target = work_dir / "retail.py"
        target.write_text(target.read_text() + "FINAL = True\n")
        kwargs["audit_events"].append(
            {"success": True, "attempt": 1, "duration_seconds": 1.0}
        )
        return "{}"

    monkeypatch.setattr(evolution, "_call_opencode", completed)
    result = evolution._run_opencode_analysis(
        prompt="batch-prompt",
        source_harness_dir=harness,
        model="provider/model",
        domain="retail",
        checkpoint_dir=checkpoint,
    )
    assert len(calls) == 1
    assert result["checkpoint_status"] == "completed"

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("completed batch call must be reused")

    monkeypatch.setattr(evolution, "_call_opencode", must_not_run)
    reused = evolution._run_opencode_analysis(
        prompt="batch-prompt",
        source_harness_dir=harness,
        model="provider/model",
        domain="retail",
        checkpoint_dir=checkpoint,
    )
    assert reused["checkpoint_status"] == "completed"


class _FakeMadClient:
    model = "fake/mad"

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
            raise RuntimeError("controlled MAD interruption")
        return f"reply-{self.calls}"

    def chat_json(self, _messages, temperature=0.0):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("controlled moderator interruption")
        return {"proposed_changes": [], "skills": []}


def test_mad_checkpoint_resumes_at_next_turn(tmp_path):
    checkpoint = tmp_path / "mad.json"
    failures = [_failure("alpha", "failure")]
    groups = {"user_stop:DB": failures}

    first = _FakeMadClient(fail_at=3)
    with pytest.raises(RuntimeError, match="MAD interruption"):
        debate.run_harness_mad(
            failures,
            groups,
            client=first,
            domain="retail",
            checkpoint_path=checkpoint,
        )
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "interrupted"
    assert state["next_turn_index"] == 2
    assert len(state["transcript"]) == 2

    resumed_client = _FakeMadClient()
    result = debate.run_harness_mad(
        failures,
        groups,
        client=resumed_client,
        domain="retail",
        checkpoint_path=checkpoint,
    )
    assert resumed_client.calls == 5  # four remaining roles plus moderator
    assert result["checkpoint_status"] == "completed"
    assert len(result["transcript"]) == 7

    completed_client = _FakeMadClient(fail_at=1)
    reused = debate.run_harness_mad(
        failures,
        groups,
        client=completed_client,
        domain="retail",
        checkpoint_path=checkpoint,
    )
    assert completed_client.calls == 0
    assert reused["checkpoint_status"] == "completed"
