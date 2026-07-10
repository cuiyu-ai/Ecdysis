"""Two-round multi-agent debate ->harness patch specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecdysis.llm_client import LLMClient

from .roles import (
    MAD_ROLES,
    ROLE_ANALYST,
    ROLE_CRITIC,
    ROLE_ENGINEER,
    ROLE_MODERATOR,
    ROLE_SYSTEM,
)

MAD_ROUNDS = 2


def _format_failure_context(
    failures: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    *,
    domain: str | None,
) -> str:
    group_lines = []
    for key, members in sorted(
        groups.items(),
        key=lambda kv: -len({m.get("task_id") for m in kv[1]}),
    ):
        task_ids = sorted({str(m.get("task_id")) for m in members})
        group_lines.append(
            f"- cluster `{key}`: {len(members)} failures, "
            f"{len(task_ids)} tasks {task_ids[:8]}"
        )
    sample = []
    for f in failures[:12]:
        sample.append(
            f"  task={f.get('task_id')} trial={f.get('trial')} "
            f"reward={f.get('reward')} term={f.get('termination_reason')} "
            f"breakdown={f.get('reward_breakdown')}"
        )
    domain_line = f"Domain harness file: `{domain}.py`\n" if domain else ""
    return (
        f"{domain_line}"
        f"## Failure clusters ({len(groups)})\n"
        + "\n".join(group_lines)
        + "\n\n## Sample failures\n"
        + "\n".join(sample)
    )


def _append_turn(
    transcript: list[dict[str, str]],
    role: str,
    content: str,
) -> None:
    transcript.append({"role": role, "content": content.strip()})


def _role_messages(
    role: str,
    context: str,
    transcript: list[dict[str, str]],
    *,
    round_num: int,
    is_final_round: bool,
) -> list[dict[str, str]]:
    if role == ROLE_MODERATOR:
        task = (
            "Produce the final harness patch specification JSON as described "
            "in your system instructions."
        )
    elif role == ROLE_ANALYST and round_num > 1:
        task = (
            "Revise your harness proposals addressing the Critic's objections. "
            "Keep changes minimal and cross-task when possible."
        )
    elif role == ROLE_CRITIC and round_num > 1:
        task = "Re-evaluate the revised proposals. Note any remaining risks."
    elif role == ROLE_ENGINEER:
        task = (
            "Summarize agreements, disputes, and what the Analyst should fix "
            "next round."
            if not is_final_round
            else "No action ->moderator follows."
        )
    else:
        task = (
            "Propose harness updates (H2/H3/H4/H5) for the failure clusters below."
        )

    history = "\n\n".join(
        f"### {t['role'].upper()}\n{t['content']}" for t in transcript[-12:]
    )
    user = (
        f"## Context\n{context}\n\n"
        f"## Debate so far\n{history or '(opening round)'}\n\n"
        f"## Your turn ({role}, round {round_num}/{MAD_ROUNDS})\n{task}"
    )
    return [
        {"role": "system", "content": ROLE_SYSTEM[role]},
        {"role": "user", "content": user},
    ]


def _parse_moderator_spec(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "failure_patterns": [],
            "proposed_changes": [],
            "skills": [],
            "implementation_notes": "",
        }
    return {
        "failure_patterns": list(parsed.get("failure_patterns") or []),
        "proposed_changes": list(parsed.get("proposed_changes") or []),
        "skills": list(parsed.get("skills") or []),
        "implementation_notes": str(parsed.get("implementation_notes") or ""),
    }


def run_harness_mad(
    failures: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    *,
    client: LLMClient,
    domain: str | None = None,
) -> dict[str, Any]:
    """Run 2-round MAD (Analyst ->Critic ->Engineer) × 2, then Moderator JSON."""
    if not failures:
        return {
            "transcript": [],
            "spec": _parse_moderator_spec(None),
            "rounds": MAD_ROUNDS,
        }

    context = _format_failure_context(failures, groups, domain=domain)
    transcript: list[dict[str, str]] = []

    for round_num in range(1, MAD_ROUNDS + 1):
        is_final = round_num == MAD_ROUNDS
        for role in MAD_ROLES:
            messages = _role_messages(
                role,
                context,
                transcript,
                round_num=round_num,
                is_final_round=is_final,
            )
            reply = client.chat(messages, temperature=0.3)
            _append_turn(transcript, role, reply)

    mod_messages = _role_messages(
        ROLE_MODERATOR,
        context,
        transcript,
        round_num=MAD_ROUNDS,
        is_final_round=True,
    )
    spec = _parse_moderator_spec(client.chat_json(mod_messages, temperature=0.0))
    _append_turn(
        transcript,
        ROLE_MODERATOR,
        json.dumps(spec, indent=2),
    )

    return {"transcript": transcript, "spec": spec, "rounds": MAD_ROUNDS}


def save_mad_transcript(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def format_spec_for_opencode(spec: dict[str, Any]) -> str:
    """Render moderator spec as instructions for the OpenCode coding agent."""
    lines = ["## MAD Consensus ->implement these harness updates\n"]
    patterns = spec.get("failure_patterns") or []
    if patterns:
        lines.append("### Failure patterns\n" + "\n".join(f"- {p}" for p in patterns))
    changes = spec.get("proposed_changes") or []
    if changes:
        lines.append("\n### Proposed changes")
        for ch in changes:
            lines.append(
                f"- [{ch.get('layer', '?')}] {ch.get('file_path', '?')}: "
                f"{ch.get('description', '')} ->{ch.get('rationale', '')}"
            )
    skills = spec.get("skills") or []
    if skills:
        lines.append(
            "\n### H5 skills (also write `evolved_skills.json` in the harness dir)"
        )
        lines.append(json.dumps({"skills": skills}, indent=2))
    notes = spec.get("implementation_notes")
    if notes:
        lines.append(f"\n### Implementation notes\n{notes}")
    lines.append(
        "\nImplement the above in the harness Python files. "
        "Keep edits minimal and test-safe. "
        "Do NOT edit base.py or append dynamic text (step counters, timestamps) "
        "to tool responses ->tasks with message_history require byte-stable replay."
    )
    return "\n".join(lines)
