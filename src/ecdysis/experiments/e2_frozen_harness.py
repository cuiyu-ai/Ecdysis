"""E2: frozen harness (H2+H3+H4+H5), no evolution. Single eval."""

from __future__ import annotations

from datetime import datetime

from ecdysis.pipeline.log import eval_output_tag, log_saved

from .base import BaseExperiment, ExperimentResult


class ExperimentE2(BaseExperiment):
    def run(self) -> ExperimentResult:
        params = self._resolve_eval_params()
        domain = params["domain"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_tag = eval_output_tag(self.label, f"frozen_harness_{timestamp}")
        result_dir = self._run_eval_subprocess(
            output_tag,
            split=params["split"],
            trials=params["trials"],
            num_tasks=params["num_tasks"],
            task_ids=params["task_ids"],
            concurrency=params["concurrency"],
            max_steps=params["max_steps"],
            agent_max_tokens=params["agent_max_tokens"],
            agent_disable_thinking=params["agent_disable_thinking"],
            user_disable_thinking=params["user_disable_thinking"],
            nl_assertions=params["nl_assertions"],
        )
        summary = {} if self.dry_run else self._load_summary(result_dir)
        result = ExperimentResult(
            experiment=self.exp_name,
            description=self.exp_config.get("description", "E2 frozen harness"),
            domain=domain,
            split=params["split"],
            model=self.exp_config.get("agent_llm"),
            user_model=self.exp_config.get("user_llm"),
            trials=params["trials"],
            timestamp=datetime.now().isoformat(),
            summary=summary,
        )
        out = self._save_result(result, domain, timestamp)
        log_saved(self.label, out)
        return result
