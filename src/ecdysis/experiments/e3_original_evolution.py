"""E3: serial per-failure evolution (Life-Harness baseline).

Pipeline (each train round):
  1. train eval on train split (1 trial per task)
  2. collect failed trajectories
  3. serial OpenCode ->one staging call per failure (original mode)
  4. merge into HarnessPatch + H5 skills, apply for next round

Final: test eval, then git revert tau2/harness.
"""

from __future__ import annotations

from typing import Any

from ecdysis.evolution import analyze_failures_serial, collect_all_trajectories
from ecdysis.harness_patch import HarnessPatch
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
)

from .base import BaseExperiment, ExperimentResult


class ExperimentE3(BaseExperiment):
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

        round_records: list[dict[str, Any]] = []
        skill_artifact_path = None
        applied_patch: HarnessPatch | None = None
        last_passk: float | None = None

        self.snapshot_harness()
        try:
            for round_num in range(1, rounds + 1):
                with timer.step("apply_patch", label=label, round_num=round_num):
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
                    )

                if self.dry_run:
                    log_round(
                        label,
                        round_num,
                        "dry-run: serial OpenCode (one call per failure) would run here",
                    )
                    analysis: dict[str, Any] = {"skills": [], "code_changes": []}
                    failures: list[dict[str, Any]] = []
                else:
                    with timer.step(
                        "collect_failures", label=label, round_num=round_num
                    ):
                        failures = self.collect_failures(train_dir)
                    if not failures:
                        log_round(label, round_num, "no failures ->skip evolution")
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
                        f"{len(failures)} failures ->serial OpenCode (per failure)",
                    )
                    with timer.step(
                        "evolution_serial", label=label, round_num=round_num
                    ):
                        analysis = analyze_failures_serial(
                            failures,
                            agent_model=agent_model,
                            all_trajectories=all_trajs,
                            domain=domain,
                        )

                pre_round_skill = skill_artifact_path
                pre_round_patch = applied_patch
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
                        evolution_mode="original",
                        metadata={"evolution_model": agent_model},
                        previous_skill_artifact=pre_round_skill,
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
            )
            return finish_evolution_result(
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
        finally:
            self.revert_harness()
