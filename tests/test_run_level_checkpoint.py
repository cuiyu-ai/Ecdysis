import json
from pathlib import Path

import pytest

from ecdysis.pipeline import steps


class _FakeExperiment:
    exp_name = "E3"
    dry_run = False

    def __init__(self, root: Path):
        self.exp_config = {
            "evolution_rounds": 3,
            "evolution_agent_model": "provider/model",
        }
        self.eval_config = {
            "domain": "retail",
            "train_split": "train",
            "test_split": "test",
            "train_trials": 1,
            "test_trials": 3,
            "evolution_checkpoint_root": str(root),
        }
        self.config_path = root / "config.yaml"

    def _resolve_eval_params(self):
        return {
            "domain": "retail",
            "split": "test",
            "trials": 3,
            "num_tasks": 20,
            "task_ids": None,
            "concurrency": 1,
            "max_steps": None,
            "agent_max_tokens": 1,
            "agent_disable_thinking": True,
            "user_disable_thinking": True,
            "nl_assertions": False,
            "train_split": "train",
            "test_split": "test",
            "train_trials": 1,
            "test_trials": 3,
        }

    def _artifact_dir(self, domain, timestamp):
        return Path("artifacts") / domain / timestamp


def test_run_state_restores_completed_round(tmp_path, monkeypatch):
    monkeypatch.setattr(
        steps,
        "snapshot_harness_files",
        lambda: {"retail.py": "BASE"},
    )
    exp = _FakeExperiment(tmp_path / "checkpoint")
    first = steps.init_evolution_run(exp)
    assert first["start_round"] == 1
    run_id = first["timestamp"]

    skill = tmp_path / "round_1.json"
    patch = tmp_path / "round_1_harness.json"
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "results.json").write_text(json.dumps({"simulations": []}))
    (train_dir / "harness_summary.json").write_text(json.dumps({}))
    first["run_state"].record_train(1, train_dir, None)
    first["run_state"].complete_round(
        1,
        record={"round": 1, "train_dir": str(train_dir)},
        skill_artifact_path=skill,
        patch_chain=[patch],
        last_passk=0.5,
        stop=False,
    )

    resumed = steps.init_evolution_run(exp)
    assert resumed["timestamp"] == run_id
    assert resumed["start_round"] == 2
    assert resumed["skill_artifact_path"] == skill
    assert resumed["patch_chain"] == [patch]
    assert resumed["last_passk"] == 0.5
    assert resumed["round_records"] == [
        {"round": 1, "train_dir": str(train_dir)}
    ]
    assert resumed["run_state"].train_dir(1) == train_dir


def test_run_state_rejects_changed_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        steps,
        "snapshot_harness_files",
        lambda: {"retail.py": "BASE"},
    )
    exp = _FakeExperiment(tmp_path / "checkpoint")
    steps.init_evolution_run(exp)
    exp.eval_config["test_trials"] = 4
    with pytest.raises(RuntimeError, match="does not match"):
        steps.init_evolution_run(exp)


def test_stop_control_is_excluded_from_run_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        steps,
        "snapshot_harness_files",
        lambda: {"retail.py": "BASE"},
    )
    exp = _FakeExperiment(tmp_path / "checkpoint")
    steps.init_evolution_run(exp)
    exp.eval_config["stop_after_evolution_round"] = 1
    resumed = steps.init_evolution_run(exp)
    assert resumed["start_round"] == 1
