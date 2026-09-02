"""Pipeline helpers for multi-round evolution experiments."""

from importlib import import_module

# Avoid eager imports here: ``experiments.base`` imports ``pipeline.log``, which
# loads this package; importing ``steps`` would circle back into ``base``.

__all__ = [
    "PipelineTimer",
    "agent_model_name",
    "append_round_record",
    "apply_patch_before_round",
    "check_early_stop",
    "finish_evolution_result",
    "init_evolution_run",
    "persist_round_artifacts",
    "run_final_test_eval",
    "run_train_eval",
]


def __getattr__(name: str):
    if name == "PipelineTimer":
        from .timing import PipelineTimer

        return PipelineTimer
    steps = import_module(".steps", __name__)

    if hasattr(steps, name):
        return getattr(steps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
