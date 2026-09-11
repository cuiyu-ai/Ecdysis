import json

from ecdysis.benchmark import run_benchmark


class _Adapter:
    name = "external-benchmark"
    version = "test-commit"

    def load_tasks(self, split):
        assert split == "test"
        return [{"id": "a"}, {"id": "b"}]

    def execute(self, harness, task):
        return {"value": harness + task["id"]}

    def encode_output(self, output):
        return output


def test_external_benchmark_runner_records_metadata_and_outputs(tmp_path):
    run = run_benchmark(_Adapter(), "frozen-", split="test")

    assert run.benchmark == "external-benchmark"
    assert run.version == "test-commit"
    assert run.result.succeeded == 2
    path = run.write_jsonl(tmp_path / "results.jsonl", encoder=_Adapter().encode_output)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["benchmark"] == "external-benchmark"
    assert row["output"] == {"value": "frozen-a"}
