"""Legacy trajectory-judge helpers (superseded by ``ecdysis.mad``).

E5 now uses Multi-Agent Debate during harness evolution, not a judge
that selects among trajectories.  This module is kept for reference only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecdysis.llm_client import OpenRouterClient
from ecdysis.prompts import DEBATE_JUDGE_PROMPT


def _format_trajectory_for_judge(
    task: dict[str, Any], sim: dict[str, Any], idx: int
) -> str:
    messages = sim.get("messages") or []
    last_messages = messages[-8:]
    transcript = []
    for message in last_messages:
        role = message.get("role")
        content = (message.get("content") or "")[:300]
        tool_calls = message.get("tool_calls") or []
        tool_str = ""
        if tool_calls:
            tool_str = " " + " | ".join(
                f"{tc.get('name')}({str(tc.get('arguments'))[:120]})"
                for tc in tool_calls[:3]
            )
        transcript.append(f"[{role}] {content}{tool_str}")
    ri = sim.get("reward_info") or {}
    reward = ri.get("reward")
    breakdown = ri.get("reward_breakdown") or {}
    return (
        f"### Trajectory {idx}\n"
        f"trial={sim.get('trial')} termination={sim.get('termination_reason')} "
        f"reward={reward} breakdown={breakdown}\n"
        f"task_instructions={str(task.get('user_scenario', {}).get('instructions', ''))[:400]}\n"
        + "\n".join(transcript)
    )


def _format_judge_request(task: dict[str, Any], sims: list[dict[str, Any]]) -> str:
    task_desc = str(task.get("user_scenario", {}).get("instructions", ""))[:800]
    trajectories_blob = "\n\n".join(
        _format_trajectory_for_judge(task, sim, i) for i, sim in enumerate(sims)
    )
    return DEBATE_JUDGE_PROMPT.format(
        task_description=task_desc, trajectories=trajectories_blob
    )


def _extract_judgement(parsed: dict[str, Any] | None, num_trajectories: int) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"selected": 0, "reasoning": "", "scores": []}
    selected = parsed.get("selected_trajectory")
    if not isinstance(selected, int) or selected < 0 or selected >= num_trajectories:
        selected = 0
    return {
        "selected": int(selected),
        "reasoning": str(parsed.get("reasoning", "")),
        "scores": parsed.get("scores", []) or [],
    }


def debate_select_best_per_task(
    result_dir: Path,
    *,
    client: OpenRouterClient,
) -> dict[str, Any]:
    """Run the debate judge over every task in the result.json.

    Returns a dict ``{task_id: {selected_index, reasoning, scores, sims}}``.
    Tasks with only a single simulation are passed through unchanged.
    """
    results_path = Path(result_dir) / "results.json"
    data = json.loads(results_path.read_text())
    tasks_by_id = {str(t.get("id")): t for t in data.get("tasks", [])}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for sim in data.get("simulations", []):
        by_task.setdefault(str(sim.get("task_id")), []).append(sim)

    verdicts: dict[str, Any] = {}
    for task_id, sims in by_task.items():
        if len(sims) <= 1:
            verdicts[task_id] = {
                "selected_index": 0,
                "reasoning": "single-trial pass-through",
                "scores": [],
                "sim_indices": list(range(len(sims))),
            }
            continue
        task = tasks_by_id.get(task_id, {})
        user_prompt = _format_judge_request(task, sims)
        messages = [
            {"role": "system", "content": "You are a strict but fair judge of LLM-agent customer-service trajectories."},
            {"role": "user", "content": user_prompt + "\n\nReturn a single JSON object with keys 'selected_trajectory', 'reasoning', 'scores'."},
        ]
        parsed = client.chat_json(messages)
        judgement = _extract_judgement(parsed, len(sims))
        verdicts[task_id] = {
            "selected_index": judgement["selected"],
            "reasoning": judgement["reasoning"],
            "scores": judgement["scores"],
            "sim_indices": list(range(len(sims))),
        }
    return {
        "by_task": verdicts,
        "num_tasks": len(by_task),
    }


def losing_simulations(
    results: dict[str, Any],
    verdicts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter simulations to the debate losers (used as evolution failures)."""
    by_task = verdicts.get("by_task", {})
    keep: list[dict[str, Any]] = []
    for sim in results.get("simulations", []):
        task_id = str(sim.get("task_id"))
        verdict = by_task.get(task_id)
        if not verdict:
            keep.append(sim)
            continue
        sims_for_task = sorted(
            (s for s in results.get("simulations", []) if str(s.get("task_id")) == task_id),
            key=lambda s: (s.get("trial") or 0),
        )
        idx = next(
            (i for i, s in enumerate(sims_for_task) if s is sim),
            None,
        )
        if idx is None or idx != verdict.get("selected_index"):
            keep.append(sim)
    return keep


def _loser_keys(losing_sims: list[dict[str, Any]]) -> set[tuple[str, int | None]]:
    return {(str(s.get("task_id")), s.get("trial")) for s in losing_sims}


def filter_failures_to_losers(
    failures: list[dict[str, Any]],
    losing_sims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop any failure whose (task_id, trial) was the debate winner."""
    losers = _loser_keys(losing_sims)
    return [f for f in failures if (f.get("task_id"), f.get("trial")) in losers]
