"""E5: Cross-Instance Learning evolution + Multi-Agent Debate (MAD) for harness updates.

Pipeline (each train round):
  1. train eval ->same ``train_trials`` as E4 (default 1 trial/task)
  2. collect all failed trajectories
  3. Cross-Instance Learning cluster failures by (termination, reward_basis)
  4. MAD ->Analyst / Critic / Engineer × 2 rounds ->Moderator patch spec
  5. OpenCode ->apply spec in staging copy ->HarnessPatch + H5 skills
  6. apply patch for next round

Final: test eval on held-out split, then revert tau2/harness via git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ecdysis.evolution import (
    analyze_failures_mad_batched,
    collect_all_trajectories,
)
from ecdysis.harness_patch import HarnessPatch, load_patch
from ecdysis.llm_client import build_client_from_exp_config
from ecdysis.pipeline.log import log_round, log_train_round_header
from ecdysis.pipeline.steps import (
    append_round_record,
    check_early_stop,
    finish_evolution_result,
    init_evolution_run,
    run_final_test_eval,
    run_train_eval,
)

from .base import BaseExperiment, DryRunClient, ExperimentResult


class ExperimentE5(BaseExperiment):
    """Cross-Instance Learning + MAD evolution; readable ``run()`` without trajectory judge."""

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

        if self.dry_run:
            mad_client = DryRunClient(
                model=self.exp_config.get("evolution_llm") or "<mad-model>"
            )
        else:
            mad_client = build_client_from_exp_config(
                self.exp_config, role="evolution"
            )

        round_records: list[dict[str, Any]] = []
        skill_artifact_path = None
        applied_patch: HarnessPatch | None = None
        last_passk: float | None = None

        self.snapshot_harness()
        try:
            for round_num in range(1, rounds + 1):
                with timer.step("apply_patch", label=label, round_num=round_num):
                    if applied_patch is not None and not self.dry_run:
                        self.apply_harness_patch(applied_patch)

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
                        "dry-run: Cross-Instance Learning cluster ->MAD (2 rounds) ->OpenCode staging would run here",
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
                        f"{len(failures)} failures ->Cross-Instance Learning cluster ->MAD (2 rounds) ->OpenCode",
                    )
                    with timer.step(
                        "evolution_mad", label=label, round_num=round_num
                    ):
                        analysis = analyze_failures_mad_batched(
                            failures,
                            mad_client=mad_client,
                            agent_model=agent_model,
                            all_trajectories=all_trajs,
                            domain=domain,
                            artifact_dir=artifact_dir,
                            round_num=round_num,
                        )

                from ecdysis.artifacts import EvolvedSkill

                pre_round_skill = skill_artifact_path
                pre_round_patch = applied_patch
                skill_dicts = analysis.get("skills", [])
                code_changes = analysis.get("code_changes", [])
                skills = [EvolvedSkill.from_dict(s) for s in skill_dicts]
                prev_skill = (
                    Path(pre_round_skill) if pre_round_skill is not None else None
                )
                with timer.step(
                    "persist_artifacts", label=label, round_num=round_num
                ):
                    skill_artifact_path = self._save_skill_artifact(
                        domain=domain,
                        round_num=round_num,
                        mode="mad",
                        skills=skills,
                        metadata={
                            "round": round_num,
                            "num_failures": len(failures) if not self.dry_run else 0,
                            "evolution_model": mad_client.model,
                            "mad_rounds": 2,
                        },
                        out_dir=artifact_dir,
                        previous_artifact=prev_skill,
                    )
                    new_patch_path = self._save_harness_patch(
                        domain=domain,
                        round_num=round_num,
                        skill_artifact_path=skill_artifact_path,
                        file_changes=code_changes,
                        metadata={"round": round_num, "mad": True},
                        out_dir=artifact_dir,
                        parent_patch=applied_patch,
                    )
                    applied_patch = (
                        load_patch(new_patch_path) if code_changes else applied_patch
                    )

                train_summary = {} if self.dry_run else self._load_summary(train_dir)
                curr_passk = train_summary.get("pass@k", 0)
                append_round_record(
                    round_records,
                    round_num=round_num,
                    train_dir=train_dir,
                    failures=failures if not self.dry_run else [],
                    analysis=analysis,
                    skill_artifact_path=skill_artifact_path,
                    new_patch_path=artifact_dir / f"round_{round_num}_harness.json",
                    pass_at_k=curr_passk,
                    timer=timer,
                    extra={
                        "mad_artifact": str(
                            artifact_dir / f"round_{round_num}_mad.json"
                        ),
                    },
                )
                log_round(
                    label,
                    round_num,
                    f"{len(skill_dicts)} skills, "
                    f"{len(code_changes)} code changes, pass@k={curr_passk}",
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
