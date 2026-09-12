"""Ecdysis core algorithms."""

from ecdysis.evolution import (
    analyze_failures_fdcr,
    collect_all_trajectories,
    collect_failed_trajectories,
    group_failures_by_pattern,
)
from ecdysis.inference import InferenceRecord, InferenceResult, InferenceRunner
from ecdysis.training import EcdysisTrainer, TrainingConfig, TrainingResult

__version__ = "0.1.0"

__all__ = [
    "analyze_failures_fdcr",
    "collect_all_trajectories",
    "collect_failed_trajectories",
    "group_failures_by_pattern",
    "InferenceRecord",
    "InferenceResult",
    "InferenceRunner",
    "EcdysisTrainer",
    "TrainingConfig",
    "TrainingResult",
]
