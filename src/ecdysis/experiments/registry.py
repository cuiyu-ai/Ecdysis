"""Registry mapping experiment names to their implementation classes."""

from __future__ import annotations

from typing import Any

from .base import BaseExperiment

_REGISTRY: dict[str, str] = {}


def _register(name: str, dotted_path: str) -> None:
    _REGISTRY[name] = dotted_path


_register("E1", "ecdysis.experiments.e1_baseline.ExperimentE1")
_register("E2", "ecdysis.experiments.e2_frozen_harness.ExperimentE2")
_register("E3", "ecdysis.experiments.e3_original_evolution.ExperimentE3")
_register("E4", "ecdysis.experiments.e4_cross_instance_evolution.ExperimentE4")
_register("E5", "ecdysis.experiments.e5_debate.ExperimentE5")


def registered_experiments() -> list[str]:
    return list(_REGISTRY)


def build_experiment(
    exp_name: str,
    exp_config: dict[str, Any],
    eval_config: dict[str, Any],
    *,
    config_path,
    dry_run: bool = False,
) -> BaseExperiment:
    if exp_name not in _REGISTRY:
        raise ValueError(
            f"Unknown experiment '{exp_name}'. Known: {registered_experiments()}"
        )
    import importlib

    module_path, _, class_name = _REGISTRY[exp_name].rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(
        exp_name=exp_name,
        exp_config=exp_config,
        eval_config=eval_config,
        config_path=config_path,
        dry_run=dry_run,
    )
