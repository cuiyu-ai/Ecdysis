"""Core failure-analysis algorithms used by Ecdysis."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ecdysis.artifacts import EvolvedSkill, SkillArtifact


def _load_results(results: dict | str | Path) -> dict:
    if isinstance(results, dict):
        return results
    path = Path(results)
    if path.is_dir():
        path = path / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("results must contain a JSON object")
    return payload


def _compact_message(message: dict[str, Any], max_content_chars: int) -> dict[str, Any]:
    compact: dict[str, Any] = {"role": message.get("role")}
    content = message.get("content")
    if content:
        compact["content"] = str(content)[:max_content_chars]

    tool_calls = message.get("tool_calls")
    if tool_calls:
        compact["tool_calls"] = [
            {
                "name": call.get("function", {}).get("name")
                or call.get("name")
                or call.get("tool_name"),
                "arguments": call.get("function", {}).get("arguments")
                or call.get("arguments"),
            }
            for call in tool_calls
            if isinstance(call, dict)
        ]
    for key in ("tool_call_id", "requestor"):
        if message.get(key):
            compact[key] = message[key]
    return compact


def _task_by_id(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(task.get("id")): task
        for task in results.get("tasks", [])
        if isinstance(task, dict)
    }


def _reward(simulation: dict[str, Any]) -> float | None:
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, dict):
        return None
    reward = reward_info.get("reward")
    return float(reward) if reward is not None else None


def _collect_trajectories(
    results: dict | str | Path,
    *,
    failures_only: bool,
    failure_threshold: float,
    max_messages: int,
    max_content_chars: int,
) -> list[dict[str, Any]]:
    if max_messages < 0:
        raise ValueError("max_messages must be non-negative")
    if max_content_chars < 1:
        raise ValueError("max_content_chars must be positive")
    if not math.isfinite(failure_threshold):
        raise ValueError("failure_threshold must be finite")

    data = _load_results(results)
    tasks = _task_by_id(data)
    trajectories: list[dict[str, Any]] = []
    for simulation in data.get("simulations", []):
        if not isinstance(simulation, dict):
            continue
        reward = _reward(simulation)
        if failures_only and (reward is None or reward >= failure_threshold):
            continue

        task_id = str(simulation.get("task_id"))
        raw_reward_info = simulation.get("reward_info")
        reward_info = raw_reward_info if isinstance(raw_reward_info, dict) else {}
        raw_messages = simulation.get("messages")
        messages = raw_messages if isinstance(raw_messages, list) else []
        selected_messages = messages[-max_messages:] if max_messages else []
        item = {
            "task_id": task_id,
            "trial": simulation.get("trial"),
            "reward": reward,
            "termination_reason": simulation.get("termination_reason"),
            "reward_breakdown": reward_info.get("reward_breakdown"),
            "task": tasks.get(task_id, {}),
            "messages": [
                _compact_message(message, max_content_chars)
                for message in selected_messages
                if isinstance(message, dict)
            ],
        }
        if failures_only:
            item.update(
                failure=True,
                reward_basis=reward_info.get("reward_basis"),
                reward_info=reward_info.get("info"),
                simulation_info=simulation.get("info"),
            )
        trajectories.append(item)
    return trajectories


def collect_failed_trajectories(
    results: dict | str | Path,
    *,
    max_messages: int = 80,
    max_content_chars: int = 2000,
    failure_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Extract unsuccessful trajectories and compact their message evidence."""
    return _collect_trajectories(
        results,
        failures_only=True,
        failure_threshold=failure_threshold,
        max_messages=max_messages,
        max_content_chars=max_content_chars,
    )


def collect_all_trajectories(
    results: dict | str | Path,
    *,
    max_messages: int = 20,
    max_content_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Extract successful and unsuccessful trajectories in a compact form."""
    return _collect_trajectories(
        results,
        failures_only=False,
        failure_threshold=1.0,
        max_messages=max_messages,
        max_content_chars=max_content_chars,
    )


def mean_trajectory_score(results: dict | str | Path) -> float:
    """Return the arithmetic mean of all available trajectory scores."""
    scores = [
        reward
        for simulation in _load_results(results).get("simulations", [])
        if isinstance(simulation, dict)
        and (reward := _reward(simulation)) is not None
    ]
    if not scores:
        raise ValueError("results contain no scored trajectories")
    return sum(scores) / len(scores)


def group_failures_by_pattern(
    failures: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group failures by termination reason and failed reward components."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        termination = str(failure.get("termination_reason") or "unknown")
        breakdown = failure.get("reward_breakdown") or {}
        failed_components = sorted(
            str(key)
            for key, value in breakdown.items()
            if value is not None and float(value) < 1.0
        )
        component = "+".join(failed_components) or "no_breakdown"
        groups[f"{termination}:{component}"].append(failure)
    return dict(
        sorted(
            groups.items(),
            key=lambda item: (
                -len({row.get("task_id") for row in item[1]}),
                -len(item[1]),
                item[0],
            ),
        )
    )


def format_failure_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    max_tasks_per_group: int = 8,
) -> str:
    """Render grouped failures as a compact, deterministic summary."""
    if not groups:
        return "(no groups)"
    rows = sorted(
        groups.items(),
        key=lambda item: (-len({row.get("task_id") for row in item[1]}), item[0]),
    )
    lines: list[str] = []
    for key, members in rows:
        task_ids = sorted({str(member.get("task_id")) for member in members})
        shown = task_ids[:max_tasks_per_group]
        suffix = "..." if len(task_ids) > max_tasks_per_group else ""
        lines.append(
            f"- `{key}` - {len(members)} failures across {len(task_ids)} "
            f"distinct tasks; task_ids={shown}{suffix}"
        )
    return "\n".join(lines)


def analyze_failures_fdcr(
    failures: list[dict[str, Any]],
    *,
    client: Any,
    scope: str | None = None,
    checkpoint_path: Path | None = None,
    refinement_passes: int = 2,
) -> dict[str, Any]:
    """Group failure evidence and run FDCR cross-role review."""
    from ecdysis.fdcr import run_harness_fdcr

    groups = group_failures_by_pattern(failures)
    review = run_harness_fdcr(
        failures,
        groups,
        client=client,
        scope=scope,
        checkpoint_path=checkpoint_path,
        rounds=refinement_passes,
    )
    return {"groups": groups, "review": review}


def artifact_from_skills(
    *,
    experiment: str,
    scope: str,
    round_num: int,
    mode: str,
    skills: list[dict[str, Any] | EvolvedSkill],
    metadata: dict[str, Any] | None = None,
) -> SkillArtifact:
    """Build a validated skill artifact from model-produced dictionaries."""
    evolved = [
        skill if isinstance(skill, EvolvedSkill) else EvolvedSkill.from_dict(skill)
        for skill in skills
    ]
    return SkillArtifact(
        experiment=experiment,
        scope=scope,
        round=round_num,
        mode=mode,
        skills=evolved,
        metadata=metadata or {},
    )


def check_convergence(
    round_records: list[dict[str, Any]],
    threshold: float = 0.005,
    window: int = 3,
) -> bool:
    """Return true after a sustained metric plateau over the selected window."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(round_records) < window:
        return False
    recent = round_records[-window:]
    deltas = [
        float(recent[index].get("score", 0))
        - float(recent[index - 1].get("score", 0))
        for index in range(1, len(recent))
    ]
    return max(deltas) < threshold


def check_regression(
    before_score: float,
    after_score: float,
    tolerance: float = 0.05,
) -> bool:
    """Return true when the score drop exceeds the accepted tolerance."""
    return (before_score - after_score) > tolerance
