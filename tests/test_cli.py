import json

from ecdysis.__main__ import main


def test_cli_summarizes_failure_groups(tmp_path, capsys):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": "one",
                        "reward_info": {
                            "reward": 0.0,
                            "reward_breakdown": {"execution": 0.0},
                        },
                        "termination_reason": "stopped",
                        "messages": [],
                    },
                    {
                        "task_id": "two",
                        "reward_info": {"reward": 1.0},
                        "messages": [],
                    },
                ]
            }
        )
    )

    assert main([str(results_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_count"] == 1
    assert payload["groups"]["stopped:execution"] == {
        "failures": 1,
        "tasks": 1,
    }
