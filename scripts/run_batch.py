#!/usr/bin/env python3
"""Batch experiment runner for Ecdysis.

Runs experiments across multiple models and collects results.

Usage:
    # Run E1+E2 on all models (quick baseline)
    python scripts/run_batch.py --exps E1 E2 --group all

    # Run E1+E2 on small models only
    python scripts/run_batch.py --exps E1 E2 --group small

    # Run with specific domain
    python scripts/run_batch.py --exps E1 E2 --group qwen3 --domain airline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for IMPORT_PATH in (PROJECT_ROOT, PROJECT_ROOT / "scripts", SRC_DIR):
    if str(IMPORT_PATH) not in sys.path:
        sys.path.insert(0, str(IMPORT_PATH))

from models import (
    ALL_MODELS,
    USER_MODEL,
    get_model_list,
)  # noqa: E402
from ecdysis.experiment_config import (  # noqa: E402
    base_experiment_config,
    build_eval_command,
    resolve_nl_assertions,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "experiments"
BATCH_EXPERIMENTS = ["E1", "E2"]


def run_single_eval(
    exp_name: str,
    model_id: str,
    model_name: str,
    domain: str,
    split: str,
    trials: int,
    num_tasks: int = None,
) -> dict:
    """Run a single evaluation and return results."""
    config = base_experiment_config(exp_name)
    if config.get("evolution_rounds"):
        raise ValueError(
            f"{exp_name} is a multi-stage evolution experiment; use "
            "scripts/run_experiment.py --config <local-run-file>.yaml instead."
        )
    config.update(
        {
            "agent_llm": model_id,
            "agent_api_base": os.getenv(
                "AGENT_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            "agent_api_key_env": os.getenv("AGENT_API_KEY_ENV", "AGENT_API_KEY"),
            "user_llm": USER_MODEL,
            "user_api_base": os.getenv(
                "USER_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            "user_api_key_env": os.getenv("USER_API_KEY_ENV", "OPENAI_API_KEY"),
        }
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_tag = f"{exp_name}_{config['name']}_{model_name}_{timestamp}"

    nl_assertions_enabled, nl_preflight = resolve_nl_assertions(
        domain,
        split,
        num_tasks=num_tasks,
        setting="auto",
    )
    cmd = build_eval_command(
        config,
        domain,
        split,
        output_tag,
        trials=trials,
        num_tasks=num_tasks,
        nl_assertions=nl_assertions_enabled,
    )

    print(f"\n{'='*60}")
    print(f"  Running: {exp_name} | {model_name} | {domain}")
    print(
        f"  NL preflight: {nl_preflight['nl_task_count']} / "
        f"{nl_preflight['selected_task_count']} selected tasks require NL; "
        f"--nl={'on' if nl_assertions_enabled else 'off'}"
    )
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    # Find results
    sim_dir = PROJECT_ROOT / "data" / "simulations" / domain
    candidates = sorted(sim_dir.glob(f"*_{output_tag}"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        print(f"  WARNING: No results found")
        return None

    results_dir = candidates[0]
    summary_file = results_dir / "harness_summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

    # Save structured result
    exp_dir = RESULTS_DIR / f"{exp_name}_{config['name']}" / domain
    exp_dir.mkdir(parents=True, exist_ok=True)

    structured_result = {
        "experiment": exp_name,
        "description": config["description"],
        "domain": domain,
        "split": split,
        "model_id": model_id,
        "model_name": model_name,
        "user_model": USER_MODEL,
        "trials": trials,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "summary": summary,
        "source_dir": str(results_dir),
    }

    result_file = exp_dir / f"{model_name}_{timestamp}.json"
    result_file.write_text(json.dumps(structured_result, indent=2))

    print(f"  ->{model_name}: pass@1={summary.get('pass@k', 'N/A')}, time={elapsed:.0f}s")

    return structured_result


def run_batch(
    experiments: list[str],
    models: list[dict],
    domain: str,
    split: str,
    trials: int,
    num_tasks: int = None,
) -> dict:
    """Run batch experiments across models."""
    all_results = {}

    total = len(experiments) * len(models)
    current = 0

    for exp_name in experiments:
        all_results[exp_name] = {}

        for model in models:
            current += 1
            model_name = model["name"]
            model_id = model["id"]

            print(f"\n{'#'*60}")
            print(f"  [{current}/{total}] {exp_name} - {model_name}")
            print(f"{'#'*60}")

            result = run_single_eval(
                exp_name=exp_name,
                model_id=model_id,
                model_name=model_name,
                domain=domain,
                split=split,
                trials=trials,
                num_tasks=num_tasks,
            )

            all_results[exp_name][model_name] = result

    return all_results


def save_batch_summary(all_results: dict, output_file: Path):
    """Save batch experiment summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    for exp_name, models in all_results.items():
        summary["experiments"][exp_name] = {}
        for model_name, result in models.items():
            if result:
                summary["experiments"][exp_name][model_name] = {
                    "pass@1": result["summary"].get("pass@k"),
                    "avg_reward": result["summary"].get("average_reward"),
                    "total_tokens": result["summary"].get("total_token_count"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved ->{output_file}")


def main():
    parser = argparse.ArgumentParser(description="Ecdysis batch experiment runner")
    parser.add_argument("--exps", nargs="+", default=["E1", "E2"],
                        choices=BATCH_EXPERIMENTS,
                        help="Experiments to run")
    parser.add_argument("--group", default="all",
                        choices=["all", "qwen3", "qwen2.5", "small", "medium", "large"],
                        help="Model group to test")
    parser.add_argument("--model", type=str, help="Specific model name (overrides --group)")
    parser.add_argument("--domain", default="airline", choices=["airline", "retail"],
                        help="Domain to evaluate")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Task split")
    parser.add_argument("--trials", type=int, default=3, help="Trials per task")
    parser.add_argument("--num-tasks", type=int, help="Limit number of tasks (for testing)")
    args = parser.parse_args()

    # Get model list
    if args.model:
        models = [m for m in ALL_MODELS if m["name"] == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not found")
            print(f"Available: {[m['name'] for m in ALL_MODELS]}")
            return 1
    else:
        models = get_model_list(args.group)

    print(f"\n{'='*60}")
    print(f"  Ecdysis Batch Experiments")
    print(f"{'='*60}")
    print(f"  Experiments: {args.exps}")
    print(f"  Models: {[m['name'] for m in models]}")
    print(f"  Domain: {args.domain}")
    print(f"  Split: {args.split}")
    print(f"  Trials: {args.trials}")
    print(f"  Num tasks: {args.num_tasks or 'all'}")
    print(f"  Total runs: {len(args.exps) * len(models)}")
    print(f"{'='*60}\n")

    # Run experiments
    all_results = run_batch(
        experiments=args.exps,
        models=models,
        domain=args.domain,
        split=args.split,
        trials=args.trials,
        num_tasks=args.num_tasks,
    )

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f"batch_summary_{timestamp}.json"
    save_batch_summary(all_results, summary_file)

    # Print results table
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<20} ", end="")
    for exp in args.exps:
        print(f"{'':>10}", end="")
    print()

    for model in models:
        model_name = model["name"]
        print(f"  {model_name:<20} ", end="")
        for exp in args.exps:
            result = all_results.get(exp, {}).get(model_name)
            if result:
                pass1 = result["summary"].get("pass@k", 0)
                print(f"{pass1:>10.3f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print(f"\n  ->Batch complete")


if __name__ == "__main__":
    sys.exit(main())
