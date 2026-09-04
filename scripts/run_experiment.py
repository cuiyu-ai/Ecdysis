#!/usr/bin/env python3
"""Unified entry point for E1–E5.

Reads a YAML config from ``configs/``, dispatches to the matching
experiment module (explicit ``run()`` pipeline), and writes summaries
under ``data/experiments/``.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _IMPORT_PATH in (_PROJECT_ROOT, _SRC_DIR):
    if str(_IMPORT_PATH) not in sys.path:
        sys.path.insert(0, str(_IMPORT_PATH))

from ecdysis.experiment_config import (  # noqa: E402
    apply_yaml_config,
    resolve_nl_assertions,
)
from ecdysis.pipeline.log import evolution_mode_summary, log_dry_run  # noqa: E402
from ecdysis.pipeline.steps import EvolutionPaused  # noqa: E402
from ecdysis.experiments import BaseExperiment, build_experiment, registered_experiments  # noqa: E402


def _print_plan(exp: BaseExperiment, eval_params: dict) -> None:
    ec = exp.exp_config
    print("=" * 60)
    print(f"Experiment: {exp.label} — {ec.get('description', '')}")
    print(f"Config     : {exp.config_path}")
    print(f"Domain     : {eval_params['domain']}")
    if ec.get("evolution_rounds"):
        print(f"Pipeline   : {evolution_mode_summary(ec)} ({ec.get('evolution_rounds')} rounds)")
        print(f"MAD        : {'enabled' if ec.get('debate') else 'disabled'}")
        print(
            f"Splits     : train={eval_params['train_split']} "
            f"(trials={eval_params['train_trials']}) → "
            f"test={eval_params['test_split']} (trials={eval_params['test_trials']})"
        )
    else:
        harness = "enabled (H2–H5)" if ec.get("harness") else "disabled"
        print(f"Harness    : {harness}")
        print(f"Split      : {eval_params['split']} (trials={eval_params['trials']})")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ecdysis unified experiment runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config (e.g. configs/E3_original_evolution.yaml)",
    )
    parser.add_argument("--domain", choices=["airline", "retail"], help="Override domain")
    parser.add_argument("--num-tasks", type=int, help="Limit number of tasks (for smoke tests)")
    parser.add_argument(
        "--rounds", type=int, help="Override evolution rounds (E3-E5 only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without running any eval or LLM call",
    )
    parser.add_argument(
        "--stop-after-evolution-round",
        type=int,
        help=(
            "Pause successfully after persisting the selected evolution round; "
            "rerun without this option (or with a larger round) to resume"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    exp = BaseExperiment.from_config_path(config_path, dry_run=args.dry_run)

    if args.domain:
        exp.eval_config["domain"] = args.domain
    if args.num_tasks is not None:
        exp.eval_config["num_tasks"] = args.num_tasks
    if args.rounds is not None:
        exp.exp_config["evolution_rounds"] = args.rounds
        if "evolution" in exp.eval_config:
            exp.eval_config["evolution"]["rounds"] = args.rounds
    if args.stop_after_evolution_round is not None:
        configured_rounds = int(exp.exp_config.get("evolution_rounds") or 0)
        if not 1 <= args.stop_after_evolution_round <= configured_rounds:
            parser.error(
                "--stop-after-evolution-round must be between 1 and "
                f"{configured_rounds}"
            )
        exp.eval_config["stop_after_evolution_round"] = (
            args.stop_after_evolution_round
        )

    eval_params = exp._resolve_eval_params()
    print(
        f"Preflight [{exp.label}]: domain={eval_params['domain']} "
        f"split={eval_params['split']} trials={eval_params['trials']} "
        f"nl={eval_params['nl_assertions']}"
    )
    _print_plan(exp, eval_params)

    if args.dry_run:
        log_dry_run(
            exp.label,
            "expanding pipeline (no eval subprocesses, LLM calls, or harness writes)",
        )

    started = datetime.now()
    try:
        result = exp.run()
    except EvolutionPaused as exc:
        elapsed = (datetime.now() - started).total_seconds()
        print(f"\n[{exp.label}] paused safely in {elapsed:.1f}s: {exc}")
        return 0
    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n[{exp.label}] done in {elapsed:.1f}s")
    timing = (result.extra or {}).get("timing")
    if timing:
        print(
            f"  train: {timing.get('train_duration_seconds', 0):.1f}s, "
            f"final test: {timing.get('final_test_duration_seconds', 0):.1f}s"
        )
    summary = result.summary or {}
    if summary:
        print("Summary:")
    for key in ("average_reward", "pass@k", "pass^k"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
