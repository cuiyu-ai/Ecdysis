"""Minimal training loop corresponding to the Ecdysis method."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from ecdysis.evolution import (
    analyze_failures_fdcr,
    collect_failed_trajectories,
    mean_trajectory_score,
)
from ecdysis.inference import InferenceResult, InferenceRunner

HarnessT = TypeVar("HarnessT")
TaskT = TypeVar("TaskT")
OutputT = TypeVar("OutputT")

type Collector[HarnessT] = Callable[[HarnessT], dict | str | Path]
type Editor[HarnessT] = Callable[
    [HarnessT, dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]],
    HarnessT,
]
type Executor[HarnessT, TaskT, OutputT] = Callable[[HarnessT, TaskT], OutputT]
type Scorer = Callable[[dict | str | Path], float]


@dataclass(frozen=True)
class TrainingConfig:
    """Controls the method-level training loop without task-specific settings."""

    rounds: int = 3
    failure_threshold: float = 1.0
    refinement_passes: int = 2
    scope: str | None = None
    checkpoint_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be at least 1")
        if self.refinement_passes < 1:
            raise ValueError("refinement_passes must be at least 1")
        if not math.isfinite(self.failure_threshold):
            raise ValueError("failure_threshold must be finite")


@dataclass(frozen=True)
class RoundRecord:
    """Auditable outcome of one evolution round."""

    round: int
    baseline_score: float
    candidate_score: float | None
    retained_score: float
    failure_count: int
    group_count: int
    accepted: bool
    status: str


@dataclass(frozen=True)
class TrainingResult[HarnessT]:
    """Frozen harness and the decisions that produced it."""

    frozen_harness: HarnessT
    final_score: float
    rounds: tuple[RoundRecord, ...] = field(default_factory=tuple)

    def infer(
        self,
        task: TaskT,
        executor: Executor[HarnessT, TaskT, OutputT],
    ) -> OutputT:
        """Execute a task with the frozen harness without further adaptation."""
        return executor(self.frozen_harness, task)

    def infer_many(
        self,
        tasks: list[TaskT],
        executor: Executor[HarnessT, TaskT, OutputT],
        *,
        fail_fast: bool = False,
    ) -> InferenceResult[TaskT, OutputT]:
        """Run a batch of tasks through the frozen harness."""
        return InferenceRunner(
            self.frozen_harness,
            executor,
        ).run(tasks, fail_fast=fail_fast)


class EcdysisTrainer[HarnessT]:
    """Run failure aggregation, FDCR, editing, and strict validation."""

    def __init__(
        self,
        *,
        collector: Collector[HarnessT],
        editor: Editor[HarnessT],
        review_client: Any,
        scorer: Scorer = mean_trajectory_score,
        config: TrainingConfig | None = None,
    ) -> None:
        self.collector = collector
        self.editor = editor
        self.review_client = review_client
        self.scorer = scorer
        self.config = config or TrainingConfig()

    def train(self, initial_harness: HarnessT) -> TrainingResult[HarnessT]:
        """Train and freeze a harness using strict score-improvement acceptance."""
        current_harness = initial_harness
        current_score: float | None = None
        records: list[RoundRecord] = []

        for round_num in range(1, self.config.rounds + 1):
            current_results = self.collector(current_harness)
            current_score = self.scorer(current_results)
            failures = collect_failed_trajectories(
                current_results,
                failure_threshold=self.config.failure_threshold,
            )
            if not failures:
                records.append(
                    RoundRecord(
                        round=round_num,
                        baseline_score=current_score,
                        candidate_score=None,
                        retained_score=current_score,
                        failure_count=0,
                        group_count=0,
                        accepted=False,
                        status="no_failures",
                    )
                )
                continue

            checkpoint_path = None
            if self.config.checkpoint_dir is not None:
                checkpoint_path = (
                    self.config.checkpoint_dir / f"round_{round_num}_fdcr.json"
                )
            analysis = analyze_failures_fdcr(
                failures,
                client=self.review_client,
                scope=self.config.scope,
                checkpoint_path=checkpoint_path,
                refinement_passes=self.config.refinement_passes,
            )
            groups = analysis["groups"]
            specification = analysis["review"]["spec"]
            candidate = self.editor(
                current_harness,
                specification,
                failures,
                groups,
            )
            candidate_score = self.scorer(self.collector(candidate))
            baseline_score = current_score
            accepted = candidate_score > baseline_score
            if accepted:
                current_harness = candidate
                current_score = candidate_score

            records.append(
                RoundRecord(
                    round=round_num,
                    baseline_score=baseline_score,
                    candidate_score=candidate_score,
                    retained_score=current_score,
                    failure_count=len(failures),
                    group_count=len(groups),
                    accepted=accepted,
                    status="accepted" if accepted else "rejected",
                )
            )

        if current_score is None:
            raise RuntimeError("training produced no score")
        return TrainingResult(
            frozen_harness=current_harness,
            final_score=current_score,
            rounds=tuple(records),
        )
