"""Named steps shared by E3–E5 evolution experiments.

Each experiment's ``run()`` calls these in order so the pipeline stays
visible in the experiment file while avoiding copy-paste drift.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ecdysis.artifacts import EvolvedSkill
from ecdysis.evolution import check_convergence, check_regression
from ecdysis.harness_patch import HarnessPatch, load_patch
from ecdysis.pipeline.log import (
    log_phase,
    log_round,
    log_saved,
    log_train_round_header,
)
from ecdysis.experiments.base import BaseExperiment, ExperimentResult
from ecdysis.pipeline.timing import PipelineTimer


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = params["domain"]
    artifact_dir = exp._artifact_dir(domain, timestamp)
    timer = PipelineTimer()
    return {
        "params": params,
        "rounds": rounds,
        "timestamp": timestamp,
        "domain": domain,
        "artifact_dir": artifact_dir,
        "agent_model": agent_model_name(exp.exp_config),
        "timer": timer,
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
) -> Path:
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
        log_round(exp.label, round_num, "regression ->revert & stop")
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
            round_records.pop()
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
    if check_convergence(round_records):
        log_round(exp.label, round_num, "converged ->stop early")
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
) -> Path:
    log_phase(label, "final", "test eval")

    def _run() -> Path:
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
            **(extra or {}),
        },
    )
    out = exp._save_result(result, domain, timestamp)
    log_saved(label, out)
    return result
