import json

from ecdysis.inference import InferenceRunner


def test_runner_records_successes_failures_and_timing(tmp_path):
    def execute(harness, task):
        if task["id"] == "bad":
            raise RuntimeError("invalid task")
        return harness + task["value"]

    result = InferenceRunner(10, execute).run(
        [{"id": "ok", "value": 2}, {"id": "bad", "value": 0}]
    )

    assert (result.total, result.succeeded, result.failed) == (2, 1, 1)
    assert result.records[0].output == 12
    assert result.records[0].task_id == "ok"
    assert result.records[0].elapsed_seconds >= 0
    assert result.records[1].error == "RuntimeError: invalid task"

    output_path = result.write_jsonl(tmp_path / "inference.jsonl")
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["output"] == 12
    assert rows[1]["error"] == "RuntimeError: invalid task"


def test_runner_fail_fast_propagates_execution_error():
    def execute(_harness, _task):
        raise ValueError("stop")

    try:
        InferenceRunner(object(), execute).run(["task"], fail_fast=True)
    except ValueError as exc:
        assert str(exc) == "stop"
    else:
        raise AssertionError("fail_fast must propagate the execution error")
