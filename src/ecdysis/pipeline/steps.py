"""Named steps shared by E3–E5 evolution experiments.

Each experiment's ``run()`` calls these in order so the pipeline stays
visible in the experiment file while avoiding copy-paste drift.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from ecdysis.artifacts import EvolvedSkill
from ecdysis.evolution import check_convergence, check_regression
from ecdysis.harness_patch import HarnessPatch, load_patch, snapshot_harness_files
from ecdysis.pipeline.log import (
    log_phase,
    log_round,
    log_saved,
    log_train_round_header,
)
from ecdysis.experiments.base import BaseExperiment, ExperimentResult
from ecdysis.pipeline.timing import PipelineTimer
from ecdysis.experiment_config import _selected_tasks


class EvolutionPaused(RuntimeError):
    """Intentional, successful pause after a completed evolution round."""


def _result_dir_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("results.json", "harness_summary.json"):
        file_path = path / name
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Checkpointed evaluation is missing {name}: {path}"
            )
        digest.update(name.encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


class EvolutionRunState:
    """Atomic run-level state shared by E3, E4, and E5."""

    def __init__(self, path: Path | None, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(self.path.suffix + ".tmp")
        pending.write_text(json.dumps(self.payload, indent=2, sort_keys=True))
        pending.replace(self.path)

    def round_state(self, round_num: int) -> dict[str, Any]:
        return (self.payload.get("rounds") or {}).get(str(round_num), {})

    def train_dir(self, round_num: int) -> Path | None:
        state = self.round_state(round_num)
        value = state.get("train_dir")
        if not value:
            return None
        path = Path(value)
        if state.get("train_fingerprint") != _result_dir_fingerprint(path):
            raise RuntimeError(
                f"Checkpointed train evaluation fingerprint mismatch: {path}"
            )
        return path

    def record_train(
        self,
        round_num: int,
        train_dir: Path,
        skill_artifact_path: Path | str | None,
    ) -> None:
        state = self.payload.setdefault("rounds", {}).setdefault(str(round_num), {})
        state.update({
            "status": "train_completed",
            "train_dir": str(train_dir),
            "train_skill_artifact": (
                str(skill_artifact_path) if skill_artifact_path else None
            ),
            "train_fingerprint": _result_dir_fingerprint(train_dir),
        })
        self.payload["status"] = "running"
        self.save()

    def complete_round(
        self,
        round_num: int,
        *,
        record: dict[str, Any],
        skill_artifact_path: Path | str | None,
        patch_chain: list[Path],
        last_passk: float | None,
        stop: bool,
    ) -> None:
        state = self.payload.setdefault("rounds", {}).setdefault(str(round_num), {})
        state.update({
            "status": "completed",
            "record": record,
            "skill_artifact": (
                str(skill_artifact_path) if skill_artifact_path else None
            ),
            "patch_chain": [str(path) for path in patch_chain],
            "last_passk": last_passk,
            "early_stop": bool(stop),
        })
        self.payload["last_completed_round"] = round_num
        self.payload["status"] = "round_completed"
        self.save()

    def final_test_dir(self) -> Path | None:
        value = self.payload.get("final_test_dir")
        if not value:
            return None
        path = Path(value)
        if self.payload.get("final_test_fingerprint") != _result_dir_fingerprint(path):
            raise RuntimeError(
                f"Checkpointed final evaluation fingerprint mismatch: {path}"
            )
        return path

    def record_final_test(self, test_dir: Path) -> None:
        self.payload["final_test_dir"] = str(test_dir)
        self.payload["final_test_fingerprint"] = _result_dir_fingerprint(test_dir)
        self.payload["status"] = "final_test_completed"
        self.save()

    def complete(self) -> None:
        self.payload["status"] = "completed"
        self.save()


def _run_fingerprint(
    exp: BaseExperiment,
    params: dict[str, Any],
    rounds: int,
) -> str:
    eval_config = {
        key: value
        for key, value in exp.eval_config.items()
        if key not in {"stop_after_evolution_round"}
    }
    payload = {
        "experiment": exp.exp_name,
        "config_path": str(exp.config_path.resolve()),
        "experiment_config": exp.exp_config,
        "evaluation_config": eval_config,
        "params": params,
        "rounds": rounds,
        "baseline_harness": snapshot_harness_files(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def agent_model_name(exp_config: dict[str, Any]) -> str:
    return (
        exp_config.get("evolution_agent_model")
        or exp_config.get("evolution_llm")
        or exp_config.get("agent_llm")
        or ""
    )


def init_evolution_run(exp: BaseExperiment) -> dict[str, Any]:
    """Common setup: eval params, artifact dir, timestamp."""
    params = exp._resolve_eval_params()
    rounds = int(exp.exp_config.get("evolution_rounds", 3))
    domain = params["domain"]
    fingerprint = _run_fingerprint(exp, params, rounds)
    checkpoint_root_value = exp.eval_config.get("evolution_checkpoint_root")
    state_path = (
        Path(str(checkpoint_root_value)).expanduser().resolve() / "run_state.json"
        if checkpoint_root_value and not exp.dry_run
        else None
    )
    state_payload: dict[str, Any] | None = None
    if state_path and state_path.exists():
        state_payload = json.loads(state_path.read_text())
        if state_payload.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Run-level checkpoint does not match the current experiment "
                f"configuration or harness baseline: {state_path}"
            )
    if state_payload is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_payload = {
            "version": 1,
            "status": "running",
            "experiment": exp.exp_name,
            "domain": domain,
            "run_id": timestamp,
            "fingerprint": fingerprint,
            "rounds": {},
            "last_completed_round": 0,
        }
    else:
        timestamp = str(state_payload["run_id"])
    run_state = EvolutionRunState(state_path, state_payload)
    run_state.save()
    artifact_dir = exp._artifact_dir(domain, timestamp)
    timer = PipelineTimer()
    completed_round = int(state_payload.get("last_completed_round") or 0)
    completed_states = [
        run_state.round_state(round_num)
        for round_num in range(1, completed_round + 1)
    ]
    round_records = [
        dict(state["record"])
        for state in completed_states
        if state.get("status") == "completed" and state.get("record")
    ]
    latest_state = completed_states[-1] if completed_states else {}
    return {
        "params": params,
        "rounds": rounds,
        "timestamp": timestamp,
        "domain": domain,
        "artifact_dir": artifact_dir,
        "agent_model": agent_model_name(exp.exp_config),
        "timer": timer,
        "run_state": run_state,
        "start_round": completed_round + 1,
        "round_records": round_records,
        "skill_artifact_path": (
            Path(latest_state["skill_artifact"])
            if latest_state.get("skill_artifact")
            else None
        ),
        "patch_chain": [
            Path(path) for path in (latest_state.get("patch_chain") or [])
        ],
        "last_passk": latest_state.get("last_passk"),
        "early_stopped": bool(latest_state.get("early_stop", False)),
    }


def apply_patch_before_round(
    exp: BaseExperiment, applied_patch: HarnessPatch | None
) -> None:
    if applied_patch is not None and not exp.dry_run:
        exp.apply_harness_patch(applied_patch)


def run_train_eval(
    exp: BaseExperiment,
    *,
    label: str,
    round_num: int,
    timestamp: str,
    params: dict[str, Any],
    skill_artifact_path: Path | str | None,
    resume_train_dir: Path | None = None,
) -> Path:
    resume_dirs = exp.eval_config.get("resume_train_dirs") or {}
    if not isinstance(resume_dirs, dict):
        raise ValueError("evaluation.resume_train_dirs must be a mapping")
    configured_resume = resume_dirs.get(round_num, resume_dirs.get(str(round_num)))
    resume_value = resume_train_dir or configured_resume
    if resume_value is not None:
        if (
            resume_train_dir is None
            and (round_num != 1 or skill_artifact_path is not None)
        ):
            raise ValueError(
                "Reusing a train evaluation is currently supported only for "
                "round 1 before any evolved artifact has been applied"
            )
        train_dir = Path(str(resume_value)).expanduser().resolve()
        results_path = train_dir / "results.json"
        summary_path = train_dir / "harness_summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                "Reusable train directory must contain results.json and "
                f"harness_summary.json: {train_dir}"
            )

        results = json.loads(results_path.read_text())
        summary = json.loads(summary_path.read_text())
        expected_tasks = _selected_tasks(
            params["domain"],
            params["train_split"],
            num_tasks=params["num_tasks"],
            task_ids=params["task_ids"],
        )
        expected_ids = {str(task.id) for task in expected_tasks}
        simulations = results.get("simulations") or []
        actual_ids = {str(sim.get("task_id")) for sim in simulations}
        expected_simulations = len(expected_ids) * int(params["train_trials"])

        mismatches: list[str] = []
        if summary.get("domain") != params["domain"]:
            mismatches.append(
                f"domain={summary.get('domain')!r}, expected {params['domain']!r}"
            )
        if actual_ids != expected_ids:
            mismatches.append(
                f"task_ids={sorted(actual_ids)}, expected {sorted(expected_ids)}"
            )
        if len(simulations) != expected_simulations:
            mismatches.append(
                f"simulations={len(simulations)}, expected {expected_simulations}"
            )
        if int(summary.get("total_simulations", -1)) != expected_simulations:
            mismatches.append(
                "harness_summary.total_simulations does not match the "
                "configured task/trial count"
            )
        if int(summary.get("infrastructure_error_count", 0)) != 0:
            mismatches.append("reused evaluation contains infrastructure errors")

        info = results.get("info") or {}
        agent_info = info.get("agent_info") or {}
        user_info = info.get("user_info") or {}
        if agent_info.get("llm") != exp.exp_config.get("agent_llm"):
            mismatches.append(
                f"agent_model={agent_info.get('llm')!r}, expected "
                f"{exp.exp_config.get('agent_llm')!r}"
            )
        if user_info.get("llm") != exp.exp_config.get("user_llm"):
            mismatches.append(
                f"user_model={user_info.get('llm')!r}, expected "
                f"{exp.exp_config.get('user_llm')!r}"
            )
        for layer in ("h2", "h3", "h4", "h5"):
            actual = bool(summary.get(f"harness_{layer}"))
            expected = bool(exp.exp_config.get(layer))
            if actual != expected:
                mismatches.append(
                    f"harness_{layer}={actual}, expected {expected}"
                )
        expected_skills = [str(skill_artifact_path)] if skill_artifact_path else []
        actual_skills = [str(item) for item in (summary.get("skill_artifacts") or [])]
        if actual_skills != expected_skills:
            mismatches.append(
                f"skill_artifacts={actual_skills}, expected {expected_skills}"
            )
        if mismatches:
            raise ValueError(
                "Reusable train evaluation is incompatible with this run: "
                + "; ".join(mismatches)
            )

        log_round(
            label,
            round_num,
            f"reusing validated train evaluation: {train_dir}",
        )
        return train_dir

    return exp._run_eval_subprocess(
        f"{label}_round{round_num}_train_{timestamp}",
        split=params["train_split"],
        trials=params["train_trials"],
        num_tasks=params["num_tasks"],
        task_ids=params["task_ids"],
        concurrency=params["concurrency"],
        max_steps=params["max_steps"],
        agent_max_tokens=params["agent_max_tokens"],
        agent_disable_thinking=params["agent_disable_thinking"],
        user_disable_thinking=params["user_disable_thinking"],
        nl_assertions=params["nl_assertions"],
        skill_artifacts=[str(skill_artifact_path)] if skill_artifact_path else None,
    )


def persist_round_artifacts(
    exp: BaseExperiment,
    *,
    domain: str,
    round_num: int,
    artifact_dir: Path,
    analysis: dict[str, Any],
    failures: list[dict[str, Any]],
    train_dir: Path,
    applied_patch: HarnessPatch | None,
    evolution_mode: str,
    metadata: dict[str, Any] | None = None,
    previous_skill_artifact: Path | str | None = None,
) -> tuple[Path, HarnessPatch | None]:
    skill_dicts = analysis.get("skills", [])
    code_changes = analysis.get("code_changes", [])
    skills = [EvolvedSkill.from_dict(s) for s in skill_dicts]
    prev_skill = (
        Path(previous_skill_artifact)
        if previous_skill_artifact is not None
        else None
    )
    skill_artifact_path = exp._save_skill_artifact(
        domain=domain,
        round_num=round_num,
        mode=evolution_mode,
        skills=skills,
        metadata={
            "round": round_num,
            "num_failures": len(failures),
            **(metadata or {}),
        },
        out_dir=artifact_dir,
        previous_artifact=prev_skill,
    )
    new_patch_path = exp._save_harness_patch(
        domain=domain,
        round_num=round_num,
        skill_artifact_path=skill_artifact_path,
        file_changes=code_changes,
        metadata={"round": round_num, **(metadata or {})},
        out_dir=artifact_dir,
        parent_patch=applied_patch,
    )
    new_applied = load_patch(new_patch_path) if code_changes else applied_patch
    return skill_artifact_path, new_applied


def append_round_record(
    round_records: list[dict[str, Any]],
    *,
    round_num: int,
    train_dir: Path,
    failures: list[dict[str, Any]],
    analysis: dict[str, Any],
    skill_artifact_path: Path | None = None,
    new_patch_path: Path | None = None,
    pass_at_k: float | None = None,
    timer: PipelineTimer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "round": round_num,
        "phase": "train",
        "train_dir": str(train_dir),
        "failures": len(failures),
    }
    if skill_artifact_path is not None:
        record["artifact"] = str(skill_artifact_path)
    if new_patch_path is not None:
        record["harness_patch"] = str(new_patch_path)
    if pass_at_k is not None:
        record["pass@k"] = pass_at_k
    if analysis:
        record["new_skills"] = len(analysis.get("skills", []))
    if analysis.get("evolution_audit"):
        record["evolution_audit"] = analysis["evolution_audit"]
    if analysis.get("skills") or analysis.get("code_changes"):
        record["patch_decision"] = {
            "status": "candidate",
            "reason": "awaiting final selection or regression check",
        }
        record["new_code_changes"] = len(analysis.get("code_changes", []))
    if timer is not None:
        record["steps"] = timer.steps_for_round(round_num)
        record["round_duration_seconds"] = timer.round_duration(round_num)
    round_records.append({**record, **(extra or {})})


def check_early_stop(
    exp: BaseExperiment,
    *,
    round_num: int,
    last_passk: float | None,
    curr_passk: float,
    round_records: list[dict[str, Any]],
    applied_patch_holder: list,
    skill_artifact_holder: list | None = None,
    pre_regression_skill: Path | str | None = None,
    pre_regression_patch: HarnessPatch | None = None,
    timer: PipelineTimer | None = None,
    label: str = "",
) -> tuple[float | None, bool]:
    """Return updated last_passk and whether the round loop should break.

    On regression, restores harness + artifacts to the pre-round state
    (before this round's evolution) so the final test eval stays consistent.
    """
    if (
        last_passk is not None
        and check_regression(last_passk, curr_passk)
        and not exp.dry_run
    ):
        log_round(exp.label, round_num, "regression — revert & stop")
        if timer is not None:
            with timer.step(
                "regression_revert",
                label=label or exp.label,
                round_num=round_num,
            ):
                exp.revert_harness()
        else:
            exp.revert_harness()
        if round_records:
            decision = round_records[-1].get("patch_decision")
            if decision is not None:
                decision.update({
                    "status": "rejected_regression",
                    "reason": (
                        f"train pass@k regressed from {last_passk} to {curr_passk}"
                    ),
                })
        applied_patch_holder.clear()
        applied_patch_holder.append(pre_regression_patch)
        if skill_artifact_holder is not None:
            skill_artifact_holder.clear()
            restored_skill = (
                Path(pre_regression_skill)
                if pre_regression_skill is not None
                else None
            )
            skill_artifact_holder.append(restored_skill)
        if pre_regression_patch is not None:
            exp.apply_harness_patch(pre_regression_patch)
        return last_passk, True
    if round_records:
        decision = round_records[-1].get("patch_decision")
        if decision is not None and decision.get("status") == "candidate":
            decision.update({
                "status": "accepted_train",
                "reason": "train evaluation did not trigger regression rollback",
            })

    if check_convergence(round_records):
        log_round(exp.label, round_num, "converged — stop early")
        return curr_passk, True
    return curr_passk, False


def run_final_test_eval(
    exp: BaseExperiment,
    *,
    label: str,
    timestamp: str,
    params: dict[str, Any],
    skill_artifact_path: Path | str | None,
    user_prompt_override: str | None = None,
    timer: PipelineTimer | None = None,
    resume_test_dir: Path | None = None,
) -> Path:
    log_phase(label, "final", "test eval")

    def _run() -> Path:
        if resume_test_dir is not None:
            results_path = resume_test_dir / "results.json"
            summary_path = resume_test_dir / "harness_summary.json"
            if not results_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(
                    f"Reusable final test is incomplete: {resume_test_dir}"
                )
            results = json.loads(results_path.read_text())
            summary = json.loads(summary_path.read_text())
            expected_tasks = _selected_tasks(
                params["domain"],
                params["test_split"],
                num_tasks=params["num_tasks"],
                task_ids=params["task_ids"],
            )
            expected_ids = {str(task.id) for task in expected_tasks}
            simulations = results.get("simulations") or []
            expected_count = len(expected_ids) * int(params["test_trials"])
            actual_ids = {str(sim.get("task_id")) for sim in simulations}
            expected_skills = (
                [str(skill_artifact_path)] if skill_artifact_path else []
            )
            actual_skills = [
                str(item) for item in (summary.get("skill_artifacts") or [])
            ]
            mismatches = []
            if summary.get("domain") != params["domain"]:
                mismatches.append("domain")
            if actual_ids != expected_ids:
                mismatches.append("task_ids")
            if len(simulations) != expected_count:
                mismatches.append("simulation_count")
            if int(summary.get("infrastructure_error_count", 0)) != 0:
                mismatches.append("infrastructure_errors")
            if actual_skills != expected_skills:
                mismatches.append("skill_artifacts")
            if mismatches:
                raise ValueError(
                    "Reusable final test is incompatible with this run: "
                    + ", ".join(mismatches)
                )
            log_phase(label, "final", f"reusing validated test: {resume_test_dir}")
            return resume_test_dir
        return exp._run_eval_subprocess(
            f"{label}_final_test_{timestamp}",
            split=params["test_split"],
            trials=params["test_trials"],
            num_tasks=params["num_tasks"],
            task_ids=params["task_ids"],
            concurrency=params["concurrency"],
            max_steps=params["max_steps"],
            agent_max_tokens=params["agent_max_tokens"],
            agent_disable_thinking=params["agent_disable_thinking"],
            user_disable_thinking=params["user_disable_thinking"],
            nl_assertions=params["nl_assertions"],
            skill_artifacts=[str(skill_artifact_path)] if skill_artifact_path else None,
            user_prompt_override=user_prompt_override,
        )

    if timer is None:
        return _run()
    with timer.step("final_test_eval", label=label, phase="test"):
        return _run()


def finish_evolution_result(
    exp: BaseExperiment,
    *,
    label: str,
    domain: str,
    timestamp: str,
    params: dict[str, Any],
    test_dir: Path,
    round_records: list[dict[str, Any]],
    artifact_dir: Path,
    extra: dict[str, Any] | None = None,
    timer: PipelineTimer | None = None,
) -> ExperimentResult:
    summary = {} if exp.dry_run else exp._load_summary(test_dir)
    timing = timer.finish() if timer is not None else None
    resume_metadata = exp.eval_config.get("resume_train_metadata") or {}
    if not isinstance(resume_metadata, dict):
        raise ValueError("evaluation.resume_train_metadata must be a mapping")
    if timing is not None and resume_metadata:
        resumed_seconds = round(
            sum(
                float(item.get("wall_seconds", 0))
                for item in resume_metadata.values()
                if isinstance(item, dict)
            ),
            3,
        )
        timing["resumed_train_duration_seconds"] = resumed_seconds
        timing["train_duration_seconds_including_resumed"] = round(
            float(timing.get("train_duration_seconds", 0)) + resumed_seconds,
            3,
        )
        timing["duration_seconds_including_resumed"] = round(
            float(timing.get("duration_seconds", 0)) + resumed_seconds,
            3,
        )
    result = ExperimentResult(
        experiment=exp.exp_name,
        description=exp.exp_config.get("description", label),
        domain=domain,
        split=params["test_split"],
        model=exp.exp_config.get("agent_llm"),
        user_model=exp.exp_config.get("user_llm"),
        trials=params["test_trials"],
        timestamp=datetime.now().isoformat(),
        summary=summary,
        rounds=round_records,
        extra={
            "artifact_dir": str(artifact_dir),
            **({"timing": timing} if timing else {}),
            **(
                {"resume_train_metadata": resume_metadata}
                if resume_metadata
                else {}
            ),
            **(extra or {}),
        },
    )
    out = exp._save_result(result, domain, timestamp)
    log_saved(label, out)
    return result
