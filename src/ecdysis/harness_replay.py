"""Message-history replay preflight for evolved harness patches.

Some benchmark tasks ship with ``initial_state.message_history``. Before each
episode, ``Environment.set_state`` re-executes mutating tools and compares
outputs to the golden ToolMessages byte-for-byte. Harness edits that change
tool response text (e.g. step counters) break this contract and surface as
``infrastructure_error`` during eval.

Evolution calls :func:`validate_staging_harness` on the OpenCode staging copy
before accepting ``code_changes``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Patterns that break EnvironmentEvaluator trajectory replay (evaluator_env.py).
_STATIC_REPLAY_HAZARDS: tuple[tuple[str, str], ...] = (
    ("_tool_call_count", "tool-call counter in harness output"),
    ("step_hint", "dynamic step suffix on tool responses"),
    ("Step {self._tool_call_count}", "Step N of M suffix"),
)

# Appended to OpenCode / MAD prompts (also mirrored in harness_design_guide.md).
REPLAY_SAFETY_GUIDE = """
## Replay safety (mandatory)

Many tasks include a pre-recorded ``message_history`` in ``initial_state``.
At episode start — and again during **environment reward evaluation** — the
environment re-executes mutating (WRITE) tools from trajectories and requires
tool outputs to match recorded ToolMessages exactly.

Therefore harness updates MUST preserve replay determinism:

- Do **NOT** edit ``base.py`` (``use_tool``, ``_serialize_tool_result``,
  ``HarnessedToolKitMixin``) unless explicitly instructed by maintainers.
- Do **NOT** append dynamic text to tool responses: step counters, session ids,
  timestamps, random tokens, or anything derived from call order / conversation
  length / READ-tool counts.
- H2 rules and H4 annotators may only depend on **DB state and tool kwargs**
  reconstructible during replay (see ``HarnessedToolKitMixin`` docstring).
- Prefer H5 skills via ``evolved_skills.json`` or minimal domain-file edits
  over changing shared harness plumbing.
- If a change might alter any mutating tool's returned string, assume it will
  fail replay and do not implement it.
""".strip()


def static_replay_hazards(changes: list[dict[str, Any]]) -> str | None:
    """Fast check for known replay-breaking edits in ``code_changes``."""
    for change in changes:
        path = str(change.get("file_path") or "")
        content = str(change.get("new_content") or "")
        if not content:
            continue
        for needle, label in _STATIC_REPLAY_HAZARDS:
            if needle in content:
                return f"{path}: {label}"
        if path == "base.py" and "def use_tool" in content:
            if "Step " in content and "max_steps" in content:
                return f"{path}: Step counter appended in use_tool"
    return None


def iter_tasks_with_message_history(domain: str) -> list[Any]:
    """Return tasks whose ``initial_state`` includes ``message_history``."""
    if domain == "airline":
        from tau2.domains.airline.environment import get_tasks

        tasks = get_tasks(None)
    elif domain == "retail":
        from tau2.domains.retail.environment import get_tasks

        tasks = get_tasks(None)
    else:
        logger.debug("replay preflight: unsupported domain %s", domain)
        return []

    out = []
    for task in tasks:
        init = getattr(task, "initial_state", None)
        hist = getattr(init, "message_history", None) if init is not None else None
        if hist:
            out.append(task)
    return out


def _build_environment(domain: str):
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment

        return get_environment(harness_enabled=True, harness_h3=True, harness_h4=True)
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment

        return get_environment(harness_enabled=True, harness_h3=True, harness_h4=True)
    raise ValueError(f"unsupported domain for replay preflight: {domain}")


def run_replay_validation(domain: str) -> list[dict[str, Any]]:
    """Replay ``set_state`` for every task with ``message_history``.

    The live ``tau2/harness/*.py`` tree must already reflect the candidate patch.
    """
    failures: list[dict[str, Any]] = []
    tasks = iter_tasks_with_message_history(domain)
    if not tasks:
        return failures

    for task in tasks:
        task_id = getattr(task, "id", None)
        init = task.initial_state
        try:
            env = _build_environment(domain)
            env.set_state(
                init.initialization_data,
                init.initialization_actions,
                list(init.message_history or []),
            )
        except Exception as exc:
            failures.append(
                {
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return failures


def _run_replay_subprocess(domain: str) -> list[dict[str, Any]]:
    script = SCRIPTS_DIR / "validate_harness_replay.py"
    result = subprocess.run(
        [sys.executable, str(script), domain],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        return []
    payload = (result.stdout or result.stderr or "").strip()
    if payload:
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return [
        {
            "task_id": None,
            "error_type": "ReplaySubprocessError",
            "error": payload or f"exit {result.returncode}",
        }
    ]


def validate_staging_harness(
    domain: str,
    staging_dir: Path,
    *,
    code_changes: list[dict[str, Any]] | None = None,
) -> str | None:
    """Apply ``staging_dir`` onto live harness, run replay checks, restore.

    Returns a human-readable error summary, or ``None`` if checks passed.
    """
    from ecdysis.harness_patch import (
        HARNESS_DIR,
        copy_harness_py_files,
        restore_harness_files,
        snapshot_harness_files,
    )

    hazard = static_replay_hazards(code_changes or [])
    if hazard is not None:
        return f"static replay hazard: {hazard}"

    if not list(staging_dir.glob("*.py")):
        return None

    backup = snapshot_harness_files()
    try:
        copy_harness_py_files(staging_dir, HARNESS_DIR)
        failures = _run_replay_subprocess(domain)
        if not failures:
            return None
        first = failures[0]
        tid = first.get("task_id")
        err = str(first.get("error", ""))[:400]
        return (
            f"message_history replay failed for {len(failures)} task(s) "
            f"(first task_id={tid}): {err}"
        )
    finally:
        restore_harness_files(backup)
