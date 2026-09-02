"""Shared scaffolding for E1-E5 experiments.

Owns the eval subprocess runner, result loading, and the H2/H3/H4
harness patch lifecycle (snapshot, apply, revert).  Subclasses only
need to implement ``run()``.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ecdysis.evolution import collect_failed_trajectories

logger = logging.getLogger(__name__)
from ecdysis.pipeline.log import (  # noqa: E402
    log_dry_run,
    log_saved,
)
from ecdysis.experiment_config import (
    apply_yaml_config,
    build_eval_command,
    resolve_nl_assertions,
)
from ecdysis.harness_patch import (
    HarnessSnapshot,
    HarnessPatch,
    apply_patch,
    revert_harness,
    save_patch,
    snapshot_harness,
)
from ecdysis.artifacts import (
    EvolvedSkill,
    SkillArtifact,
    load_skill_artifact,
    merge_skill_artifacts,
    save_skill_artifact,
)
from ecdysis.provenance import (
    sanitize_for_persistence,
    sanitize_json_file,
    write_run_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMULATIONS_DIR = PROJECT_ROOT / "data" / "simulations"
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
EVOLVED_DIR = PROJECT_ROOT / "data" / "evolved_harness"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


@dataclass
class DryRunClient:
    """Minimal client facade used to expand experiment plans without API keys."""

    model: str

    def chat(self, *args, **kwargs) -> str:
        raise RuntimeError("Dry-run client cannot make LLM calls")

    def chat_json(self, *args, **kwargs) -> dict | None:
        raise RuntimeError("Dry-run client cannot make LLM calls")


@dataclass
class ExperimentResult:
    """Structured summary written to ``data/experiments/<exp>/<domain>/``."""

    experiment: str
    description: str
    domain: str
    split: str
    model: str | None
    user_model: str | None
    trials: int
    timestamp: str
    summary: dict[str, Any] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "description": self.description,
            "domain": self.domain,
            "split": self.split,
            "model": self.model,
            "user_model": self.user_model,
            "trials": self.trials,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "rounds": self.rounds,
            **self.extra,
        }


class BaseExperiment:
    """Common eval/patch/result plumbing for E1-E5."""

    def __init__(
        self,
        exp_name: str,
        exp_config: dict[str, Any],
        eval_config: dict[str, Any],
        *,
        config_path: Path,
        dry_run: bool = False,
    ) -> None:
        self.exp_name = exp_name
        self.exp_config = exp_config
        self.eval_config = eval_config
        self.config_path = config_path
        self.dry_run = dry_run
        self._harness_snapshot: HarnessSnapshot | None = None

    @property
    def label(self) -> str:
        """Short experiment id used in logs and eval output tags (E1–E5)."""
        return self.exp_name

    @classmethod
    def from_config_path(
        cls, config_path: Path, *, dry_run: bool = False
    ) -> "BaseExperiment":
        from ecdysis.experiments.registry import build_experiment

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        import yaml

        yaml_config = yaml.safe_load(config_path.read_text())
        exp_name, exp_config, eval_config = apply_yaml_config(yaml_config)
        return build_experiment(
            exp_name,
            exp_config,
            eval_config,
            config_path=config_path,
            dry_run=dry_run,
        )

    def run(self) -> ExperimentResult:
        raise NotImplementedError

    def _resolve_eval_params(
        self, split: str | None = None, trials: int | None = None
    ) -> dict[str, Any]:
        ec = self.eval_config
        domain = ec.get("domain", "airline")
        train_split = ec.get("train_split", ec.get("split", "train"))
        test_split = ec.get("test_split", "test")
        if split is None:
            split = ec.get("split", test_split)
        if trials is None:
            trials = int(ec.get("trials", 3))
        nl_setting = ec.get("nl_assertions", "auto")
        splits_for_nl = {split}
        if self.exp_config.get("evolution_rounds"):
            splits_for_nl.update({train_split, test_split})
        if self.dry_run:
            normalized_nl = str(nl_setting).strip().lower()
            nl_enabled = normalized_nl in {"on", "true", "yes", "1", "enabled"}
        else:
            nl_enabled = False
            nl_info: dict[str, Any] = {}
            for nl_split in sorted(splits_for_nl):
                enabled, info = resolve_nl_assertions(
                    domain,
                    nl_split,
                    num_tasks=ec.get("num_tasks"),
                    task_ids=ec.get("task_ids"),
                    setting=nl_setting,
                )
                nl_info[nl_split] = info
                nl_enabled = nl_enabled or enabled
        return {
            "domain": domain,
            "split": split,
            "trials": trials,
            "num_tasks": ec.get("num_tasks"),
            "task_ids": ec.get("task_ids"),
            "concurrency": ec.get("concurrency"),
            "max_steps": ec.get("max_steps"),
            "agent_max_tokens": ec.get("agent_max_tokens"),
            "agent_disable_thinking": bool(ec.get("agent_disable_thinking", False)),
            "user_disable_thinking": bool(ec.get("user_disable_thinking", False)),
            "nl_assertions": nl_enabled,
            "train_split": train_split,
            "test_split": test_split,
            "train_trials": int(ec.get("train_trials", 1)),
            "test_trials": int(ec.get("test_trials", ec.get("trials", 3))),
        }

    def _run_eval_subprocess(
        self,
        output_tag: str,
        *,
        split: str,
        trials: int,
        num_tasks: int | None,
        task_ids: list | None,
        concurrency: int | None,
        max_steps: int | None,
        agent_max_tokens: int | None,
        agent_disable_thinking: bool,
        user_disable_thinking: bool,
        nl_assertions: bool,
        skill_artifacts: list[str] | None = None,
        user_prompt_override: str | None = None,
        timeout: int = 7200,
        max_retries: int = 3,
        backoff: float = 15.0,
    ) -> Path:
        cmd = build_eval_command(
            self.exp_config,
            self.eval_config.get("domain", "airline"),
            split,
            output_tag,
            trials=trials,
            num_tasks=num_tasks,
            task_ids=task_ids,
            concurrency=concurrency,
            max_steps=max_steps,
            agent_max_tokens=agent_max_tokens,
            agent_disable_thinking=agent_disable_thinking,
            user_disable_thinking=user_disable_thinking,
            nl_assertions=nl_assertions,
            skill_artifacts=skill_artifacts,
            user_prompt_override=user_prompt_override,
        )
        print(" ".join(shlex.quote(part) for part in cmd))
        if self.dry_run:
            return Path(f"<dry-run>/{self.eval_config.get('domain', 'airline')}/{output_tag}")
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    result_dir = self._locate_result_dir(
                        output_tag, self.eval_config.get("domain", "airline")
                    )
                    sanitize_json_file(result_dir / "results.json")
                    return result_dir
                last_exc = RuntimeError(
                    f"eval_harness.py failed for {output_tag} (exit {result.returncode})\n"
                    f"stdout (tail): {result.stdout[-2000:]}\n"
                    f"stderr (tail): {result.stderr[-2000:]}"
                )
            except subprocess.TimeoutExpired as exc:
                last_exc = exc
            if attempt >= max_retries:
                break
            logger.warning(
                "eval subprocess failed (attempt %d/%d), retrying in %.0fs",
                attempt, max_retries, backoff * attempt,
            )
            time.sleep(backoff * attempt)
        raise RuntimeError(
            f"eval_harness.py failed after {max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _locate_result_dir(output_tag: str, domain: str) -> Path:
        sim_dir = SIMULATIONS_DIR / domain
        candidates = sorted(
            sim_dir.glob(f"*_{output_tag}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No simulation results found for {output_tag} under {sim_dir}"
            )
        return candidates[0]

    @staticmethod
    def _load_results(result_dir: Path) -> dict:
        results_path = result_dir / "results.json"
        return json.loads(results_path.read_text())

    @staticmethod
    def _load_summary(result_dir: Path) -> dict:
        summary_path = result_dir / "harness_summary.json"
        if not summary_path.exists():
            return {}
        return json.loads(summary_path.read_text())

    def collect_failures(self, result_dir: Path) -> list[dict[str, Any]]:
        results = self._load_results(result_dir)
        return collect_failed_trajectories(results)

    def _experiment_dir(self, domain: str) -> Path:
        slug = f"{self.exp_name}_{self.exp_config.get('name', self.exp_name.lower())}"
        return EXPERIMENTS_DIR / slug / domain

    def _save_result(
        self,
        result: ExperimentResult,
        domain: str,
        timestamp: str,
    ) -> Path:
        out_dir = self._experiment_dir(domain)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{timestamp}.json"
        payload = sanitize_for_persistence(result.to_dict())
        manifest_path = write_run_manifest(
            project_root=PROJECT_ROOT,
            experiment=self.exp_name,
            experiment_config=self.exp_config,
            evaluation_config=self.eval_config,
            config_path=self.config_path,
            domain=domain,
            run_id=timestamp,
            dry_run=self.dry_run,
            result_payload=payload,
        )
        payload["provenance"] = {"manifest": str(manifest_path)}
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path

    def _artifact_dir(
        self, domain: str, timestamp: str, slug_suffix: str = ""
    ) -> Path:
        slug = f"{self.exp_name}_{self.exp_config.get('name', self.exp_name.lower())}"
        return EVOLVED_DIR / slug / domain / f"{timestamp}{slug_suffix}"

    def _save_skill_artifact(
        self,
        *,
        domain: str,
        round_num: int,
        mode: str,
        skills: list[EvolvedSkill],
        metadata: dict[str, Any],
        out_dir: Path,
        previous_artifact: Path | None = None,
    ) -> Path:
        if previous_artifact and previous_artifact.exists():
            parents = [load_skill_artifact(previous_artifact)]
        else:
            parents = []
        merged = merge_skill_artifacts(
            experiment=self.exp_name,
            domain=domain,
            round_num=round_num,
            mode=mode,
            artifacts=parents,
            new_skills=skills,
            metadata=metadata,
        )
        out_path = out_dir / f"round_{round_num}.json"
        save_skill_artifact(merged, out_path)
        return out_path

    def _save_harness_patch(
        self,
        *,
        domain: str,
        round_num: int,
        skill_artifact_path: Path | None,
        file_changes: list[dict[str, Any]],
        metadata: dict[str, Any],
        out_dir: Path,
        parent_patch: Path | None = None,
    ) -> Path:
        from ecdysis.harness_patch import HarnessFileChange

        changes = [
            HarnessFileChange(
                file_path=fc["file_path"],
                new_content=fc["new_content"],
                description=fc.get("description", ""),
                target=fc.get("target", ""),
            )
            for fc in file_changes
        ]
        patch = HarnessPatch(
            experiment=self.exp_name,
            domain=domain,
            round=round_num,
            file_changes=changes,
            parent_patch=str(parent_patch) if parent_patch else None,
            skill_artifact=str(skill_artifact_path) if skill_artifact_path else None,
            metadata=metadata,
        )
        out_path = out_dir / f"round_{round_num}_harness.json"
        save_patch(patch, out_path)
        return out_path

    def snapshot_harness(self) -> None:
        if self.dry_run:
            log_dry_run(self.label, "skipping harness snapshot")
            return
        if self._harness_snapshot is not None:
            raise RuntimeError("Harness snapshot already exists for this experiment")
        self._harness_snapshot = snapshot_harness(project_root=PROJECT_ROOT)

    def apply_harness_patch(self, patch: HarnessPatch) -> None:
        apply_patch(patch, project_root=PROJECT_ROOT)

    def revert_harness(self, *, finalize: bool = False) -> None:
        if self.dry_run:
            log_dry_run(self.label, "skipping harness revert")
            return
        if self._harness_snapshot is None:
            raise RuntimeError("Cannot revert harness: no experiment snapshot exists")
        try:
            revert_harness(self._harness_snapshot, project_root=PROJECT_ROOT)
        finally:
            if finalize:
                self._harness_snapshot.cleanup()
                self._harness_snapshot = None
