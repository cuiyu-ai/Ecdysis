"""Harness evolution helpers for E3/E4.

Deterministic helpers (failure extraction, coarse grouping) live next
to the LLM-driven analyzers that produce H5 skills and H2/H3/H4
harness patches for each round.

The LLM analysis uses OpenCode CLI (open-source coding agent) to
iterate on harness code with file tools.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ecdysis.artifacts import EvolvedSkill, SkillArtifact
from ecdysis.harness_replay import REPLAY_SAFETY_GUIDE, validate_staging_harness

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESIGN_GUIDE_PATH = PROJECT_ROOT / "harness_design_guide.md"


def _load_results(results: dict | str | Path) -> dict:
    if isinstance(results, dict):
        return results
    path = Path(results)
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text())


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    content = message.get("content")
    compact: dict[str, Any] = {"role": role}
    if content:
        compact["content"] = str(content)[:2000]
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
        ]
    if message.get("tool_call_id"):
        compact["tool_call_id"] = message.get("tool_call_id")
    if message.get("requestor"):
        compact["requestor"] = message.get("requestor")
    return compact


def _task_by_id(results: dict) -> dict[str, dict]:
    return {str(task.get("id")): task for task in results.get("tasks", [])}


def _reward(simulation: dict[str, Any]) -> float | None:
    reward_info = simulation.get("reward_info")
    if reward_info is None:
        return None
    reward = reward_info.get("reward")
    return float(reward) if reward is not None else None


def collect_failed_trajectories(
    results: dict | str | Path,
    *,
    max_messages: int = 80,
) -> list[dict[str, Any]]:
    data = _load_results(results)
    tasks = _task_by_id(data)
    failures: list[dict[str, Any]] = []

    for simulation in data.get("simulations", []):
        reward = _reward(simulation)
        if reward == 1.0:
            continue
        task_id = str(simulation.get("task_id"))
        reward_info = simulation.get("reward_info") or {}
        messages = simulation.get("messages") or []
        failures.append(
            {
                "task_id": task_id,
                "trial": simulation.get("trial"),
                "reward": reward,
                "termination_reason": simulation.get("termination_reason"),
                "reward_basis": reward_info.get("reward_basis"),
                "reward_breakdown": reward_info.get("reward_breakdown"),
                "reward_info": reward_info.get("info"),
                "simulation_info": simulation.get("info"),
                "task": tasks.get(task_id, {}),
                "messages": [
                    _compact_message(message)
                    for message in messages[-max_messages:]
                    if isinstance(message, dict)
                ],
            }
        )
    return failures


def collect_all_trajectories(
    results: dict | str | Path,
    *,
    max_messages: int = 20,
) -> list[dict[str, Any]]:
    """Collect all trajectories (success + failure) for the agent to inspect."""
    data = _load_results(results)
    tasks = _task_by_id(data)
    trajectories: list[dict[str, Any]] = []

    for simulation in data.get("simulations", []):
        reward = _reward(simulation)
        task_id = str(simulation.get("task_id"))
        reward_info = simulation.get("reward_info") or {}
        messages = simulation.get("messages") or []
        trajectories.append(
            {
                "task_id": task_id,
                "trial": simulation.get("trial"),
                "reward": reward,
                "termination_reason": simulation.get("termination_reason"),
                "reward_breakdown": reward_info.get("reward_breakdown"),
                "task": tasks.get(task_id, {}),
                "messages": [
                    _compact_message(message)
                    for message in messages[-max_messages:]
                    if isinstance(message, dict)
                ],
            }
        )
    return trajectories


def group_failures_by_pattern(
    failures: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        termination = failure.get("termination_reason") or "unknown"
        breakdown = failure.get("reward_breakdown") or {}
        failed_basis = [
            str(key)
            for key, value in breakdown.items()
            if value is not None and float(value) < 1.0
        ]
        basis = "+".join(sorted(failed_basis)) if failed_basis else "no_breakdown"
        groups[f"{termination}:{basis}"].append(failure)
    return dict(groups)


def artifact_from_skills(
    *,
    experiment: str,
    domain: str,
    round_num: int,
    mode: str,
    skills: list[dict[str, Any] | EvolvedSkill],
    metadata: dict[str, Any] | None = None,
) -> SkillArtifact:
    evolved = [
        skill if isinstance(skill, EvolvedSkill) else EvolvedSkill.from_dict(skill)
        for skill in skills
    ]
    return SkillArtifact(
        experiment=experiment,
        domain=domain,
        round=round_num,
        mode=mode,
        skills=evolved,
        metadata=metadata or {},
    )


def _format_trajectory_compact(trajectory: dict[str, Any]) -> str:
    task_id = trajectory.get("task_id")
    reward = trajectory.get("reward")
    termination = trajectory.get("termination_reason")
    breakdown = trajectory.get("reward_breakdown") or {}
    messages = trajectory.get("messages") or []
    compact_messages = []
    for message in messages[-8:]:
        role = message.get("role")
        content = (message.get("content") or "")[:300]
        tool_calls = message.get("tool_calls") or []
        tool_str = ""
        if tool_calls:
            tool_str = " " + " | ".join(
                f"{tc.get('name')}({str(tc.get('arguments'))[:150]})"
                for tc in tool_calls[:3]
            )
        compact_messages.append(f"[{role}] {content}{tool_str}")
    status = "SUCCESS" if reward == 1.0 else "FAILED"
    return (
        f"[{status}] task_id={task_id} reward={reward} termination={termination} "
        f"breakdown={breakdown}\n"
        + "\n".join(compact_messages)
    )


def _read_harness_dir() -> Path:
    return PROJECT_ROOT / "tau2" / "harness"


def _copy_harness_tree(source_dir: Path, dest_dir: Path) -> None:
    """Copy harness ``*.py`` files into ``dest_dir`` (OpenCode sandbox)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob("*.py"):
        shutil.copy2(path, dest_dir / path.name)


class _OpenCodeWorkspace:
    """Temporary harness copy so OpenCode never edits ``tau2/harness`` in place."""

    def __init__(self, source_dir: Path) -> None:
        staging_root = PROJECT_ROOT / "data" / "evolved_harness"
        staging_root.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="Ecdysis-opencode-",
            dir=staging_root,
        )
        self.path = Path(self._tmpdir.name)
        _copy_harness_tree(source_dir, self.path)

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *args) -> None:
        self.close()


def _snapshot_harness_files(harness_dir: Path) -> dict[str, str]:
    snapshot = {}
    for f in harness_dir.glob("*.py"):
        snapshot[f.name] = f.read_text()
    return snapshot


def _diff_harness_files(
    before: dict[str, str], harness_dir: Path
) -> list[dict[str, Any]]:
    changes = []
    for f in harness_dir.glob("*.py"):
        old = before.get(f.name, "")
        new = f.read_text()
        if old != new:
            changes.append(
                {
                    "file_path": f.name,
                    "target": "harness_code",
                    "description": f"Modified {f.name}",
                    "new_content": new,
                }
            )
    return changes


def _load_agent_skills(work_dir: Path) -> list[dict[str, Any]]:
    skills_file = work_dir / "evolved_skills.json"
    if not skills_file.exists():
        return []
    try:
        data = json.loads(skills_file.read_text())
        if isinstance(data, list):
            return data
        return data.get("skills", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _load_design_guide() -> str:
    if DESIGN_GUIDE_PATH.exists():
        return DESIGN_GUIDE_PATH.read_text()
    return ""


def _format_groups_for_prompt(
    groups: dict[str, list[dict[str, Any]]],
    *,
    max_tasks_per_group: int = 8,
) -> str:
    """Render failure groups as a compact cross-task summary for the Cross-Instance Learning prompt.

    Sorted by number of distinct task ids so cross-task clusters surface first.
    """
    if not groups:
        return "(no groups)"
    rows = sorted(
        groups.items(),
        key=lambda kv: -len({m.get("task_id") for m in kv[1]}),
    )
    lines: list[str] = []
    for key, members in rows:
        unique_tasks = sorted({str(m.get("task_id")) for m in members})
        task_count = len(unique_tasks)
        sample_term = members[0].get("termination_reason") or "unknown"
        sample_breakdown = members[0].get("reward_breakdown") or {}
        shown = unique_tasks[:max_tasks_per_group]
        suffix = "..." if len(unique_tasks) > max_tasks_per_group else ""
        lines.append(
            f"- `{key}` ->{len(members)} failures across {task_count} distinct "
            f"tasks (sample termination={sample_term}, sample "
            f"breakdown={sample_breakdown}); task_ids={shown}{suffix}"
        )
    return "\n".join(lines)


def format_failure_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    max_tasks_per_group: int = 8,
) -> str:
    """Render grouped failures for logs, tests, or lightweight reports."""
    return _format_groups_for_prompt(
        groups, max_tasks_per_group=max_tasks_per_group
    )


def _filter_replay_safe_changes(
    changes: list[dict[str, Any]],
    *,
    domain: str | None,
    staging_dir: Path,
) -> list[dict[str, Any]]:
    """Drop ``code_changes`` that fail message_history replay preflight."""
    if not changes or not domain:
        return changes
    err = validate_staging_harness(
        domain, staging_dir, code_changes=changes
    )
    if err is None:
        return changes
    logger.warning(
        "Replay preflight rejected %d harness file change(s) for domain=%s: %s",
        len(changes),
        domain,
        err,
    )
    return []


def _build_agent_prompt(
    trajectories: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    mode: str,
    harness_dir: Path,
    domain: str | None = None,
) -> str:
    design_guide = _load_design_guide()
    harness_files = list(harness_dir.glob("*.py"))
    harness_listing = ", ".join(f.name for f in harness_files)

    trajectory_blob = "\n\n---\n\n".join(
        _format_trajectory_compact(t) for t in trajectories[:30]
    )

    if mode == "cross_instance":
        groups = group_failures_by_pattern(failures)
        grouping_section = (
            "## Cross-Task Failure Grouping\n"
            "Failures have been clustered by "
            "`(termination_reason, failed_reward_basis)`. Groups containing -> "
            "distinct task IDs are cross-task patterns and should be the primary "
            "unit of generalization. Design harness updates that fix the cluster, "
            "not the individual case.\n\n"
            f"{_format_groups_for_prompt(groups)}"
        )
        task_framing = (
            "Identify recurring failure patterns **across tasks**. Prioritize "
            "groups with multiple distinct task IDs; treat single-task failures "
            "as supporting evidence but not as the primary unit for harness "
            "updates."
        )
    else:
        grouping_section = (
            "## Per-Task Focus\n"
            "Each failure is analyzed in isolation. Look for patterns that "
            "recur across this task's multiple trials but do not assume "
            "cross-task similarity."
        )
        task_framing = (
            "Analyze each failure in isolation. Propose targeted harness "
            "updates that fix this specific failure mode; do not generalize "
            "to other tasks."
        )

    domain_hint = ""
    if domain:
        domain_hint = (
            f"\nPrimary domain harness file for this run: `{domain}.py`. "
            "Prefer minimal edits there unless a shared helper in another file "
            "is clearly required.\n"
        )

    return f"""You are a coding agent responsible for improving a runtime harness for a deterministic LLM-agent environment. Your goal is to improve task performance by adapting the runtime interface between the frozen model and the environment, without changing model weights, benchmark tasks, or environment evaluation logic.

## Design Guide
{design_guide}

{REPLAY_SAFETY_GUIDE}

## Current Harness Implementation
The harness code is in the current directory: {harness_listing}
{domain_hint}

{grouping_section}

## Trajectories (Previous Iteration)
{trajectory_blob}

## Your Task
{task_framing}

For each pattern, determine the earliest lifecycle point where it can be reliably detected or prevented: before interaction, during task conditioning, before environment execution, or after execution.

Focus on mechanically identifiable deterministic failures such as invalid action formats, wrong tool conventions, missing required fields, repeated no-op actions, loops, premature submissions, budget exhaustion, or recurring procedural mistakes.

Directly implement targeted, minimal updates in the appropriate harness layer. Do not only return an analysis report. Do not use hidden oracle information, test labels, task modifications, environment transition changes, or evaluation-criteria changes.

After editing, run or recommend the narrowest regression checks available. Inspect cases where the harness may over-trigger, block a valid action, inject misleading guidance, or reduce performance on previously successful trajectories.

When finished, summarize:
1. dominant failure patterns found;
2. harness layer responsible for each update;
3. implemented code changes;
4. why each update is safe under the deterministic environment contract;
5. remaining failure modes to monitor next.
"""


def _call_opencode(
    prompt: str,
    work_dir: Path,
    model: str = "dashscope/deepseek-v4-pro",
    timeout: int = 600,
    max_retries: int = 3,
    backoff: float = 10.0,
) -> str:
    """Call OpenCode CLI to run the coding agent. Retries on transient
    failures (non-zero exit, subprocess.TimeoutExpired) with linear
    backoff. Permanent failures (4xx in stderr) are not retried.
    """
    cmd = [
        "opencode",
        "run",
        "--dir", str(work_dir),
        "--model", model,
        "--format", "json",
        prompt,
    ]
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
            )
            if result.returncode == 0:
                return result.stdout
            last_exc = RuntimeError(
                f"OpenCode CLI failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
        if attempt >= max_retries:
            break
        logger.warning(
            "OpenCode failed (attempt %d/%d), retrying in %.0fs",
            attempt, max_retries, backoff * attempt,
        )
        time.sleep(backoff * attempt)
    raise RuntimeError(
        f"OpenCode CLI failed after {max_retries} attempts: {last_exc}"
    )


def _run_opencode_analysis(
    *,
    prompt: str,
    source_harness_dir: Path,
    model: str,
    domain: str | None = None,
) -> dict[str, Any]:
    """Run OpenCode in an isolated copy of the current harness tree."""
    with _OpenCodeWorkspace(source_harness_dir) as workspace:
        before = _snapshot_harness_files(workspace)
        _call_opencode(prompt, workspace, model=model)
        changes = _diff_harness_files(before, workspace)
        changes = _filter_replay_safe_changes(
            changes, domain=domain, staging_dir=workspace
        )
        skills = _load_agent_skills(workspace)
        skills_file = workspace / "evolved_skills.json"
        if skills_file.exists():
            skills_file.unlink()
    return {"skills": skills, "code_changes": changes}


def analyze_failures_serial(
    failures: list[dict[str, Any]],
    *,
    agent_model: str | None = None,
    all_trajectories: list[dict[str, Any]] | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """E3-style serial analysis. One OpenCode CLI call per failure."""
    if not failures:
        return {"skills": [], "code_changes": []}
    live_harness = _read_harness_dir()
    model = agent_model or "dashscope/deepseek-v4-pro"
    trajectories = all_trajectories or failures

    all_skills: list[dict[str, Any]] = []
    with _OpenCodeWorkspace(live_harness) as workspace:
        initial = _snapshot_harness_files(workspace)
        for failure in failures:
            prompt = _build_agent_prompt(
                trajectories,
                [failure],
                mode="original",
                harness_dir=workspace,
                domain=domain,
            )
            _call_opencode(prompt, workspace, model=model)
            all_skills.extend(_load_agent_skills(workspace))
            skills_file = workspace / "evolved_skills.json"
            if skills_file.exists():
                skills_file.unlink()
        changes = _diff_harness_files(initial, workspace)
        changes = _filter_replay_safe_changes(
            changes, domain=domain, staging_dir=workspace
        )

    return {"skills": all_skills, "code_changes": changes}


def analyze_failures_batched(
    failures: list[dict[str, Any]],
    *,
    agent_model: str | None = None,
    all_trajectories: list[dict[str, Any]] | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """E4-style batched analysis. Single OpenCode CLI call over all grouped failures."""
    if not failures:
        return {"skills": [], "code_changes": []}
    live_harness = _read_harness_dir()
    model = agent_model or "dashscope/deepseek-v4-pro"
    trajectories = all_trajectories or failures

    prompt = _build_agent_prompt(
        trajectories,
        failures,
        mode="cross_instance",
        harness_dir=live_harness,
        domain=domain,
    )
    return _run_opencode_analysis(
        prompt=prompt,
        source_harness_dir=live_harness,
        model=model,
        domain=domain,
    )


def analyze_failures_mad_batched(
    failures: list[dict[str, Any]],
    *,
    mad_client,
    agent_model: str | None = None,
    all_trajectories: list[dict[str, Any]] | None = None,
    domain: str | None = None,
    artifact_dir: Path | None = None,
    round_num: int = 1,
) -> dict[str, Any]:
    """E5: Cross-Instance Learning clustering -> 2-round MAD -> OpenCode staging."""
    if not failures:
        return {"skills": [], "code_changes": [], "mad": None}

    from ecdysis.mad.debate import (
        format_spec_for_opencode,
        run_harness_mad,
        save_mad_transcript,
    )

    groups = group_failures_by_pattern(failures)
    mad_result = run_harness_mad(
        failures, groups, client=mad_client, domain=domain
    )
    if artifact_dir is not None:
        save_mad_transcript(
            mad_result, artifact_dir / f"round_{round_num}_mad.json"
        )

    live_harness = _read_harness_dir()
    model = agent_model or "dashscope/deepseek-v4-pro"
    trajectories = all_trajectories or failures

    base_prompt = _build_agent_prompt(
        trajectories,
        failures,
        mode="cross_instance",
        harness_dir=live_harness,
        domain=domain,
    )
    mad_section = format_spec_for_opencode(mad_result["spec"])
    prompt = f"{mad_section}\n\n---\n\n{base_prompt}"

    opencode_out = _run_opencode_analysis(
        prompt=prompt,
        source_harness_dir=live_harness,
        model=model,
        domain=domain,
    )
    opencode_out["mad"] = mad_result
    return opencode_out


def check_convergence(
    round_records: list[dict[str, Any]],
    threshold: float = 0.005,
    window: int = 3,
) -> bool:
    """Check if evolution has converged.

    Requires at least ``window`` rounds of history. Returns True when the
    *maximum* pass@k improvement observed in the last ``window`` rounds is
    below ``threshold``. Using the max (not the last delta) avoids being
    fooled by single-round noise ->only a sustained plateau triggers a
    stop.
    """
    if len(round_records) < window:
        return False
    recent = round_records[-window:]
    deltas = [
        recent[i].get("pass@k", 0) - recent[i - 1].get("pass@k", 0)
        for i in range(1, len(recent))
    ]
    if not deltas:
        return False
    return max(deltas) < threshold


def check_regression(
    before_passk: float,
    after_passk: float,
    tolerance: float = 0.05,
) -> bool:
    """Check if harness changes caused regression (pass@k dropped > tolerance)."""
    return (before_passk - after_passk) > tolerance
