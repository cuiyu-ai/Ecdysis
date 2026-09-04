"""Shared experiment configuration and command-building helpers.

The files in ``scripts/`` are CLI entry points.  This module holds the reusable
pieces they share: default experiment definitions, YAML config merging,
evaluation command construction, and NL-assertion preflight checks.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(
    os.getenv("ECDYSIS_RUNTIME_ROOT", str(PROJECT_ROOT))
).expanduser()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"

if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))


def _load_project_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
        except ModuleNotFoundError:
            return

        load_dotenv(env_file)


_load_project_env()


AGENT_API_BASE = os.getenv(
    "AGENT_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
AGENT_API_KEY = (
    os.getenv("AGENT_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
USER_API_BASE = os.getenv(
    "USER_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
AGENT_API_KEY_ENV = os.getenv("AGENT_API_KEY_ENV", "AGENT_API_KEY")
USER_API_KEY_ENV = os.getenv("USER_API_KEY_ENV", "OPENAI_API_KEY")
USER_LLM = os.getenv("USER_LLM", "openai/deepseek-v4-flash")
AGENT_LLM = os.getenv("AGENT_LLM", "openai/qwen3-8b")

if AGENT_API_KEY and not os.getenv("AGENT_API_KEY"):
    os.environ["AGENT_API_KEY"] = AGENT_API_KEY
if os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")


DEFAULT_EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "E1": {
        "name": "baseline",
        "harness": False,
        "h2": False,
        "h3": False,
        "h4": False,
        "h5": False,
        "description": "No harness baseline",
    },
    "E2": {
        "name": "frozen_harness",
        "harness": True,
        "h2": True,
        "h3": True,
        "h4": True,
        "h5": True,
        "h5_top_k": 1,
        "description": "Frozen harness (no evolution)",
    },
    "E3": {
        "name": "original_evolution",
        "harness": True,
        "h2": True,
        "h3": True,
        "h4": True,
        "h5": True,
        "h5_top_k": 1,
        "evolution_rounds": 3,
        "evolution_mode": "original",
        "description": "Serial, single-task evolution (3 rounds)",
    },
    "E4": {
        "name": "cross_instance_evolution",
        "harness": True,
        "h2": True,
        "h3": True,
        "h4": True,
        "h5": True,
        "h5_top_k": 1,
        "evolution_rounds": 3,
        "evolution_mode": "cross_instance",
        "description": "Mixed Training batch evolution",
    },
    "E5": {
        "name": "debate",
        "harness": True,
        "h2": True,
        "h3": True,
        "h4": True,
        "h5": True,
        "h5_top_k": 1,
        "evolution_rounds": 3,
        "evolution_mode": "cross_instance",
        "debate": True,
        "description": "Mixed Training + MAD harness evolution (2-round multi-agent debate)",
    },
}


def experiment_names() -> list[str]:
    return list(DEFAULT_EXPERIMENT_CONFIGS)


def base_experiment_config(exp_name: str) -> dict[str, Any]:
    if exp_name not in DEFAULT_EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {exp_name}")
    return copy.deepcopy(DEFAULT_EXPERIMENT_CONFIGS[exp_name])


def get_experiment_dir(
    exp_name: str,
    domain: str,
    exp_config: dict[str, Any] | None = None,
) -> Path:
    config = exp_config or DEFAULT_EXPERIMENT_CONFIGS[exp_name]
    return EXPERIMENTS_DIR / f"{exp_name}_{config['name']}" / domain


def _get_config_value(config: dict[str, Any], key: str, default=None):
    value = config.get(key, default)
    return default if value is None else value


def _parse_nl_assertions(value) -> str:
    """Normalize the NL assertion setting to auto/on/off."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"on", "true", "yes", "1", "enabled"}:
        return "on"
    if normalized in {"off", "false", "no", "0", "disabled"}:
        return "off"
    raise ValueError(
        "evaluation.nl_assertions must be one of: auto, true/on, false/off"
    )


def _load_tasks(domain: str, split: str):
    if domain == "airline":
        from tau2.domains.airline.environment import get_tasks
    elif domain == "retail":
        from tau2.domains.retail.environment import get_tasks
    elif domain == "telecom":
        from tau2.domains.telecom.environment import get_tasks
    else:
        from tau2.domains.banking_knowledge.environment import get_tasks
    return get_tasks(split)


def _selected_tasks(
    domain: str,
    split: str,
    num_tasks: int | None = None,
    task_ids: list | None = None,
):
    tasks = _load_tasks(domain, split)
    if task_ids:
        id_set = {str(tid) for tid in task_ids}
        tasks = [task for task in tasks if task.id in id_set]
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def _task_requires_nl(task) -> bool:
    criteria = getattr(task, "evaluation_criteria", None)
    reward_basis = getattr(criteria, "reward_basis", None) or []
    return any(
        str(basis).endswith("NL_ASSERTION")
        or getattr(basis, "value", None) == "nl_assertion"
        for basis in reward_basis
    )


def resolve_nl_assertions(
    domain: str,
    split: str,
    num_tasks: int | None = None,
    task_ids: list | None = None,
    setting="auto",
) -> tuple[bool, dict[str, Any]]:
    """Decide whether --nl should be enabled for the exact selected task set."""
    mode = _parse_nl_assertions(setting)
    tasks = _selected_tasks(domain, split, num_tasks=num_tasks, task_ids=task_ids)
    nl_task_ids = [task.id for task in tasks if _task_requires_nl(task)]
    requires_nl = bool(nl_task_ids)

    if mode == "on" and not requires_nl:
        raise ValueError(
            "NL assertions were enabled, but the selected task set does not "
            "include any NL_ASSERTION reward basis. Do not enable --nl for "
            "this run."
        )
    if mode == "off" and requires_nl:
        preview = ", ".join(nl_task_ids[:10])
        suffix = "..." if len(nl_task_ids) > 10 else ""
        raise ValueError(
            "The selected task set requires NL assertion evaluation, but "
            f"evaluation.nl_assertions is off. NL task ids: {preview}{suffix}"
        )

    return (
        requires_nl if mode == "auto" else mode == "on",
        {
            "mode": mode,
            "selected_task_count": len(tasks),
            "requires_nl": requires_nl,
            "nl_task_count": len(nl_task_ids),
            "nl_task_ids": nl_task_ids,
        },
    )


def apply_yaml_config(
    yaml_config: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return isolated experiment and evaluation configs from a YAML document."""
    exp_name = yaml_config.get("experiment", {}).get("name")
    if not exp_name:
        raise ValueError("YAML config is missing experiment.name")

    exp_config = base_experiment_config(exp_name)

    experiment = yaml_config.get("experiment", {}) or {}
    if experiment.get("description"):
        exp_config["description"] = experiment["description"]

    model = yaml_config.get("model", {}) or {}
    model_fields = {
        "agent": "agent_llm",
        "agent_api_base": "agent_api_base",
        "agent_api_key_env": "agent_api_key_env",
        "user": "user_llm",
        "user_api_base": "user_api_base",
        "user_api_key_env": "user_api_key_env",
        "evolution": "evolution_llm",
        "evolution_api_base": "evolution_api_base",
        "evolution_api_key_env": "evolution_api_key_env",
        "judge": "judge_llm",
        "judge_api_base": "judge_api_base",
        "judge_api_key_env": "judge_api_key_env",
        "evolution_agent": "evolution_agent_model",
        "evolution_agent_api_base": "evolution_agent_api_base",
        "evolution_agent_api_key_env": "evolution_agent_api_key_env",
    }
    for yaml_key, config_key in model_fields.items():
        if model.get(yaml_key):
            exp_config[config_key] = model[yaml_key]

    harness = yaml_config.get("harness", {}) or {}
    if harness:
        exp_config["harness"] = bool(
            harness.get("enabled", exp_config.get("harness", False))
        )
        for key in ("h2", "h3", "h4", "h5"):
            if key in harness:
                exp_config[key] = bool(harness[key])
        if "h5_top_k" in harness:
            exp_config["h5_top_k"] = harness["h5_top_k"]

    evolution = yaml_config.get("evolution", {}) or {}
    if evolution:
        if evolution.get("enabled", False):
            if "rounds" in evolution:
                exp_config["evolution_rounds"] = int(evolution["rounds"])
            mode = evolution.get("mode")
            if mode:
                # Preserve current public mode names without embedding legacy aliases.
                normalized_mode = str(mode).strip()
                exp_config["evolution_mode"] = normalized_mode
        else:
            exp_config.pop("evolution_rounds", None)
            exp_config.pop("evolution_mode", None)

    debate = yaml_config.get("debate", {}) or {}
    if debate:
        if debate.get("enabled", False):
            exp_config["debate"] = True
        else:
            exp_config.pop("debate", None)

    return exp_name, exp_config, dict(yaml_config.get("evaluation", {}) or {})


def build_eval_command(
    exp_config: dict[str, Any],
    domain: str,
    split: str,
    output_tag: str,
    trials: int = 1,
    num_tasks: int | None = None,
    task_ids: list | None = None,
    concurrency: int | None = None,
    max_steps: int | None = None,
    agent_max_tokens: int | None = None,
    agent_disable_thinking: bool = False,
    user_disable_thinking: bool = False,
    nl_assertions: bool = False,
    skill_artifacts: list[str] | None = None,
    user_prompt_override: str | None = None,
) -> list[str]:
    """Build the eval_harness.py command."""
    agent_llm = _get_config_value(exp_config, "agent_llm", AGENT_LLM)
    user_llm = _get_config_value(exp_config, "user_llm", USER_LLM)
    agent_api_base = _get_config_value(exp_config, "agent_api_base", AGENT_API_BASE)
    agent_api_key_env = _get_config_value(
        exp_config, "agent_api_key_env", AGENT_API_KEY_ENV
    )
    user_api_base = _get_config_value(exp_config, "user_api_base", USER_API_BASE)
    user_api_key_env = _get_config_value(
        exp_config, "user_api_key_env", USER_API_KEY_ENV
    )

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "eval_harness.py"),
        "--domain",
        domain,
        "--split",
        split,
        "--trials",
        str(trials),
        "--agent-llm",
        agent_llm,
        "--agent-api-base",
        agent_api_base,
        "--agent-api-key-env",
        agent_api_key_env,
        "--user-llm",
        user_llm,
    ]

    if user_api_base:
        cmd.extend(["--user-api-base", user_api_base])
    if user_api_key_env:
        cmd.extend(["--user-api-key-env", user_api_key_env])

    cmd.extend(["--output", f"{domain}/{output_tag}"])

    if exp_config["harness"]:
        cmd.append("--enabled")
        if exp_config.get("h2"):
            cmd.append("--h2")
        if exp_config.get("h3"):
            cmd.append("--h3")
        if exp_config.get("h4"):
            cmd.append("--h4")
        if exp_config.get("h5"):
            cmd.extend(["--h5", "--h5-top-k", str(exp_config.get("h5_top_k", 1))])

    if num_tasks is not None:
        cmd.extend(["--num-tasks", str(num_tasks)])
    if task_ids:
        cmd.extend(["--task-ids"] + [str(t) for t in task_ids])
    if concurrency is not None:
        cmd.extend(["--concurrency", str(concurrency)])
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if agent_max_tokens is not None:
        cmd.extend(["--agent-max-tokens", str(agent_max_tokens)])
    if agent_disable_thinking:
        cmd.append("--agent-disable-thinking")
    if user_disable_thinking:
        cmd.append("--user-disable-thinking")
    if nl_assertions:
        cmd.append("--nl")
    for artifact in skill_artifacts or []:
        cmd.extend(["--skill-artifact", str(artifact)])

    if user_prompt_override:
        cmd.extend(["--user-prompt-override", str(user_prompt_override)])

    return cmd
