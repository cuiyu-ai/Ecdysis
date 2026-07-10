"""Shared log format for E1-E5 experiment pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def eval_output_tag(label: str, *parts: str) -> str:
    """Build a simulation output tag, e.g. ``E4_round1_train_20260101_120000``."""
    return "_".join((label, *parts))


def log_train_round_header(
    label: str,
    round_num: int,
    rounds: int,
    *,
    trials: int,
    artifact: Path | str | None,
) -> None:
    print(
        f"\n[{label} round {round_num}/{rounds}] "
        f"train (trials={trials}, artifact={artifact})"
    )


def log_round(label: str, round_num: int, message: str) -> None:
    print(f"  [{label} round {round_num}] {message}")


def log_phase(label: str, phase: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"\n[{label} {phase}]{suffix}")


def log_saved(label: str, path: Path | str) -> None:
    print(f"  [{label}] saved ->{path}")


def log_note(label: str, message: str) -> None:
    print(f"  [{label}] {message}")


def log_step_timing(
    label: str,
    round_num: int | None,
    step: str,
    duration_seconds: float,
) -> None:
    where = f"round {round_num} " if round_num is not None else ""
    print(f"  [{label} {where}{step}] {duration_seconds:.1f}s")


def log_dry_run(label: str, message: str) -> None:
    print(f"  [{label} dry-run] {message}")


def evolution_mode_summary(exp_config: dict[str, Any]) -> str:
    """Human-readable evolution stack for preflight logs."""
    if not exp_config.get("evolution_rounds"):
        return "-"
    mode = exp_config.get("evolution_mode") or "-"
    if exp_config.get("debate"):
        mode = f"{mode} + MAD"
    return mode
