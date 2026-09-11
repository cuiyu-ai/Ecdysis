"""Failure-Driven Collaborative Refinement for update specifications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ecdysis.llm_client import LLMClient

from .roles import (
    FDCR_ROLES,
    ROLE_ANALYST,
    ROLE_CRITIC,
    ROLE_ENGINEER,
    ROLE_MODERATOR,
    ROLE_SYSTEM,
)

FDCR_ROUNDS = 2


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True))
    pending.replace(path)


def _format_failure_context(
    failures: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    *,
    scope: str | None,
) -> str:
    group_lines = []
    for key, members in sorted(
        groups.items(),
        key=lambda kv: -len({m.get("task_id") for m in kv[1]}),
    ):
        task_ids = sorted({str(m.get("task_id")) for m in members})
        evidence_role = "recurring" if len(task_ids) >= 2 else "auxiliary"
        group_lines.append(
            f"- cluster `{key}`: {len(members)} failures, "
            f"{len(task_ids)} tasks {task_ids[:8]} ({evidence_role} evidence)"
        )
    sample = []
    for f in failures[:12]:
        sample.append(
            f"  task={f.get('task_id')} trial={f.get('trial')} "
            f"reward={f.get('reward')} term={f.get('termination_reason')} "
            f"breakdown={f.get('reward_breakdown')}"
        )
    scope_line = f"Task scope: `{scope}`\n" if scope else ""
    return (
        f"{scope_line}"
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
    total_rounds: int,
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
            else "No action — moderator follows."
        )
    else:
        task = (
            "Propose minimal runtime updates for the failure clusters below."
        )

    history = "\n\n".join(
        f"### {t['role'].upper()}\n{t['content']}" for t in transcript[-12:]
    )
    user = (
        f"## Context\n{context}\n\n"
        f"## Review so far\n{history or '(opening round)'}\n\n"
        f"## Your turn ({role}, round {round_num}/{total_rounds})\n{task}"
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


def run_harness_fdcr(
    failures: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    *,
    client: LLMClient,
    scope: str | None = None,
    checkpoint_path: Path | None = None,
    rounds: int = FDCR_ROUNDS,
) -> dict[str, Any]:
    """Run FDCR review passes, then synthesize a moderator JSON spec."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    usage_start = (
        client.usage_event_count()
        if hasattr(client, "usage_event_count")
        else None
    )
    if not failures:
        return {
            "transcript": [],
            "spec": _parse_moderator_spec(None),
            "rounds": rounds,
            "usage": (
                client.usage_summary(start_index=usage_start)
                if usage_start is not None
                else None
            ),
        }

    context = _format_failure_context(failures, groups, scope=scope)
    context_hash = hashlib.sha256(f"{rounds}\0{context}".encode()).hexdigest()
    model = str(getattr(client, "model", ""))
    state: dict[str, Any] = {
        "version": 1,
        "status": "pending",
        "context_hash": context_hash,
        "model": model,
        "rounds": rounds,
        "next_turn_index": 0,
        "transcript": [],
        "usage_segments": [],
    }
    if checkpoint_path and checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text())
        if state.get("context_hash") != context_hash or state.get("model") != model:
            raise RuntimeError(
                f"FDCR checkpoint input/model mismatch: {checkpoint_path}"
            )
        if state.get("status") == "completed":
            return {
                "transcript": list(state.get("transcript") or []),
                "spec": _parse_moderator_spec(state.get("spec")),
                "rounds": rounds,
                "usage": state.get("usage"),
                "usage_segments": list(state.get("usage_segments") or []),
                "checkpoint_status": "completed",
            }

    transcript: list[dict[str, str]] = list(state.get("transcript") or [])
    turn_plan = [
        (round_num, role)
        for round_num in range(1, rounds + 1)
        for role in FDCR_ROLES
    ]
    next_turn = int(state.get("next_turn_index") or 0)

    for turn_index in range(next_turn, len(turn_plan)):
        round_num, role = turn_plan[turn_index]
        is_final = round_num == rounds
        state["status"] = "running"
        state["active_turn"] = {
            "index": turn_index,
            "round": round_num,
            "role": role,
        }
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, state)
        try:
            messages = _role_messages(
                role,
                context,
                transcript,
                round_num=round_num,
                total_rounds=rounds,
                is_final_round=is_final,
            )
            usage_before = (
                client.usage_event_count()
                if hasattr(client, "usage_event_count")
                else None
            )
            reply = client.chat(messages, temperature=0.3)
            _append_turn(transcript, role, reply)
            if usage_before is not None:
                state.setdefault("usage_segments", []).append(
                    client.usage_summary(start_index=usage_before)
                )
        except Exception as exc:
            state["status"] = "interrupted"
            state["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            }
            state["transcript"] = transcript
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, state)
            raise
        state["transcript"] = transcript
        state["next_turn_index"] = turn_index + 1
        state.pop("error", None)
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, state)

    try:
        mod_messages = _role_messages(
            ROLE_MODERATOR,
            context,
            transcript,
            round_num=rounds,
            total_rounds=rounds,
            is_final_round=True,
        )
        usage_before = (
            client.usage_event_count()
            if hasattr(client, "usage_event_count")
            else None
        )
        spec = _parse_moderator_spec(
            client.chat_json(mod_messages, temperature=0.0)
        )
        _append_turn(
            transcript,
            ROLE_MODERATOR,
            json.dumps(spec, indent=2),
        )
        if usage_before is not None:
            state.setdefault("usage_segments", []).append(
                client.usage_summary(start_index=usage_before)
            )
    except Exception as exc:
        state["status"] = "interrupted"
        state["active_turn"] = {"role": ROLE_MODERATOR}
        state["error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        state["transcript"] = transcript
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, state)
        raise

    usage = (
        client.usage_summary(start_index=usage_start)
        if usage_start is not None
        else None
    )
    state.update({
        "status": "completed",
        "transcript": transcript,
        "spec": spec,
        "usage": usage,
    })
    state.pop("active_turn", None)
    state.pop("error", None)
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, state)

    return {
        "transcript": transcript,
        "spec": spec,
        "rounds": rounds,
        "usage": usage,
        "usage_segments": list(state.get("usage_segments") or []),
        "checkpoint_status": "completed",
    }


def save_fdcr_transcript(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def format_update_spec(spec: dict[str, Any]) -> str:
    """Render the moderator specification as implementation instructions."""
    lines = ["## FDCR Review - implement these harness updates\n"]
    patterns = spec.get("failure_patterns") or []
    if patterns:
        lines.append("### Failure patterns\n" + "\n".join(f"- {p}" for p in patterns))
    changes = spec.get("proposed_changes") or []
    if changes:
        lines.append("\n### Proposed changes")
        for ch in changes:
            lines.append(
                f"- [{ch.get('component', '?')}] {ch.get('description', '')}: "
                f"{ch.get('rationale', '')}"
            )
    skills = spec.get("skills") or []
    if skills:
        lines.append(
            "\n### Reusable skills"
        )
        lines.append(json.dumps({"skills": skills}, indent=2))
    notes = spec.get("implementation_notes")
    if notes:
        lines.append(f"\n### Implementation notes\n{notes}")
    lines.append(
        "\nImplement only the agreed changes. Keep edits minimal and test-safe."
    )
    return "\n".join(lines)
