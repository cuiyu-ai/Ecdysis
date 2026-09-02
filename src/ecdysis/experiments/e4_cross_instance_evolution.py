"""E4: Mixed Training batch evolution (cross-task pattern discovery).

Pipeline (each train round):
  1. train eval on train split (1 trial per task)
  2. collect failed trajectories
  3. Mixed Training cluster by (termination, reward_basis)
  4. single OpenCode call in staging — batched cross-task analysis
  5. HarnessPatch + H5 skills, apply for next round

Final: test eval, then git revert tau2/harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ecdysis.evolution import analyze_failures_batched, collect_all_trajectories
from ecdysis.harness_patch import HarnessPatch, load_patch
from ecdysis.pipeline.log import log_round, log_train_round_header
from ecdysis.pipeline.steps import (
    append_round_record,
    apply_patch_before_round,
    check_early_stop,
    finish_evolution_result,
    init_evolution_run,
    persist_round_artifacts,
    run_final_test_eval,
    run_train_eval,
    EvolutionPaused,
)

from .base import BaseExperiment, ExperimentResult


class ExperimentE4(BaseExperiment):
    def run(self) -> ExperimentResult:
        label = self.label
        ctx = init_evolution_run(self)
        params = ctx["params"]
        domain = ctx["domain"]
        rounds = ctx["rounds"]
        timestamp = ctx["timestamp"]
        artifact_dir = ctx["artifact_dir"]
        agent_model = ctx["agent_model"]
        timer = ctx["timer"]

        run_state = ctx["run_state"]
        start_round = ctx["start_round"]
        round_records: list[dict[str, Any]] = ctx["round_records"]
        skill_artifact_path = ctx["skill_artifact_path"]
        patch_chain: list[Path] = ctx["patch_chain"]
        applied_patch: HarnessPatch | None = (
            load_patch(patch_chain[-1]) if patch_chain else None
        )
        last_passk: float | None = ctx["last_passk"]

        self.snapshot_harness()
        try:
            stop_after = self.eval_config.get("stop_after_evolution_round")
            if stop_after is not None and start_round > int(stop_after):
                raise EvolutionPaused(
                    f"evolution round {stop_after} was already checkpointed"
                )
            if patch_chain and not self.dry_run:
                for patch_path in patch_chain:
                    self.apply_harness_patch(load_patch(patch_path))
            round_iter = (
                [] if ctx["early_stopped"] else range(start_round, rounds + 1)
            )
            for round_num in round_iter:
                with timer.step("apply_patch", label=label, round_num=round_num):
                    if not (round_num == start_round and patch_chain):
                        apply_patch_before_round(self, applied_patch)

                log_train_round_header(
                    label,
                    round_num,
                    rounds,
                    trials=params["train_trials"],
                    artifact=skill_artifact_path,
                )
                with timer.step("train_eval", label=label, round_num=round_num):
                    train_dir = run_train_eval(
                        self,
                        label=label,
                        round_num=round_num,
                        timestamp=timestamp,
                        params=params,
                        skill_artifact_path=skill_artifact_path,
                        resume_train_dir=run_state.train_dir(round_num),
                    )
                if not self.dry_run:
                    run_state.record_train(
                        round_num, train_dir, skill_artifact_path
                    )

                if self.dry_run:
                    log_round(
                        label,
                        round_num,
                        "dry-run: Mixed Training batch OpenCode (one call, cross-task) would run here",
                    )
                    analysis: dict[str, Any] = {"skills": [], "code_changes": []}
                    failures: list[dict[str, Any]] = []
                else:
                    with timer.step(
                        "collect_failures", label=label, round_num=round_num
                    ):
                        failures = self.collect_failures(train_dir)
                    if not failures:
                        log_round(label, round_num, "no failures — skip evolution")
                        append_round_record(
                            round_records,
                            round_num=round_num,
                            train_dir=train_dir,
                            failures=[],
                            analysis={},
                            timer=timer,
                        )
                        continue

                    with timer.step(
                        "collect_trajectories", label=label, round_num=round_num
                    ):
                        all_trajs = collect_all_trajectories(train_dir)
                    log_round(
                        label,
                        round_num,
                        f"{len(failures)} failures → Mixed Training cluster → batch OpenCode",
                    )
                    with timer.step("evolution_cross_instance", label=label, round_num=round_num):
                        checkpoint_root = Path(
                            self.eval_config.get(
                                "evolution_checkpoint_root",
                                artifact_dir / "batch_checkpoints",
                            )
                        ).expanduser()
                        analysis = analyze_failures_batched(
                            failures,
                            agent_model=agent_model,
                            all_trajectories=all_trajs,
                            domain=domain,
                            checkpoint_dir=(
                                checkpoint_root / f"round_{round_num}"
                            ).resolve(),
                            timeout=int(
                                self.eval_config.get(
                                    "evolution_timeout_seconds", 600
                                )
                            ),
                            max_retries=int(
                                self.eval_config.get(
                                    "evolution_max_retries", 1
                                )
                            ),
                        )

                pre_round_skill = skill_artifact_path
                analysis["evolution_audit"] = {
                    "mode": "batched",
                    "opencode_calls_requested": 1 if failures else 0,
                    "mad_calls_requested": 0,
                    "opencode_usage": analysis.get("opencode_usage"),
                }

                pre_round_patch = applied_patch
                pre_round_patch_chain = list(patch_chain)
                with timer.step(
                    "persist_artifacts", label=label, round_num=round_num
                ):
                    skill_artifact_path, applied_patch = persist_round_artifacts(
                        self,
                        domain=domain,
                        round_num=round_num,
                        artifact_dir=artifact_dir,
                        analysis=analysis,
                        failures=failures,
                        train_dir=train_dir,
                        applied_patch=applied_patch,
                        evolution_mode="cross_instance",
                        metadata={"evolution_model": agent_model},
                        previous_skill_artifact=pre_round_skill,
                    )
                if analysis.get("code_changes"):
                    patch_chain.append(
                        artifact_dir / f"round_{round_num}_harness.json"
                    )

                train_summary = {} if self.dry_run else self._load_summary(train_dir)
                curr_passk = train_summary.get("pass@k", 0)
                append_round_record(
                    round_records,
                    round_num=round_num,
                    train_dir=train_dir,
                    failures=failures,
                    analysis=analysis,
                    skill_artifact_path=skill_artifact_path,
                    new_patch_path=artifact_dir / f"round_{round_num}_harness.json",
                    pass_at_k=curr_passk,
                    timer=timer,
                )
                log_round(
                    label,
                    round_num,
                    f"{len(analysis.get('skills', []))} skills, "
                    f"{len(analysis.get('code_changes', []))} code changes, "
                    f"pass@k={curr_passk}",
                )

                patch_box: list = [applied_patch]
                skill_box: list = [skill_artifact_path]
                last_passk, stop = check_early_stop(
                    self,
                    round_num=round_num,
                    last_passk=last_passk,
                    curr_passk=curr_passk,
                    round_records=round_records,
                    applied_patch_holder=patch_box,
                    skill_artifact_holder=skill_box,
                    pre_regression_skill=pre_round_skill,
                    pre_regression_patch=pre_round_patch,
                    timer=timer,
                    label=label,
                )
                applied_patch = patch_box[0]
                skill_artifact_path = skill_box[0]
                if (
                    round_records
                    and round_records[-1].get("patch_decision", {}).get("status")
                    == "rejected_regression"
                ):
                    patch_chain = pre_round_patch_chain
                if not self.dry_run:
                    run_state.complete_round(
                        round_num,
                        record=round_records[-1],
                        skill_artifact_path=skill_artifact_path,
                        patch_chain=patch_chain,
                        last_passk=last_passk,
                        stop=stop,
                    )
                if (
                    stop_after == round_num
                ):
                    raise EvolutionPaused(
                        f"completed and checkpointed evolution round {round_num}"
                    )
                if stop:
                    break

            if applied_patch is not None and not self.dry_run:
                with timer.step("apply_final_patch", label=label, phase="setup"):
                    self.apply_harness_patch(applied_patch)

            test_dir = run_final_test_eval(
                self,
                label=label,
                timestamp=timestamp,
                params=params,
                skill_artifact_path=skill_artifact_path,
                timer=timer,
                resume_test_dir=run_state.final_test_dir(),
            )
            if not self.dry_run:
                run_state.record_final_test(test_dir)
            result = finish_evolution_result(
                self,
                label=label,
                domain=domain,
                timestamp=timestamp,
                params=params,
                test_dir=test_dir,
                round_records=round_records,
                artifact_dir=artifact_dir,
                timer=timer,
            )
            if not self.dry_run:
                run_state.complete()
            return result
        finally:
            self.revert_harness(finalize=True)
