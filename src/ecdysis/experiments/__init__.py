"""Per-experiment code for E1-E5.

Each experiment implements its own ``run()`` with an explicit pipeline.
Shared eval / patch / artifact helpers live in ``base.py``; named
evolution steps live in ``ecdysis.pipeline.steps``.
"""

from .base import BaseExperiment, ExperimentResult
from .registry import build_experiment, registered_experiments

__all__ = [
    "BaseExperiment",
    "ExperimentResult",
    "build_experiment",
    "registered_experiments",
]
