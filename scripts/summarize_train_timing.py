#!/usr/bin/env python3
"""Summarize step timings from E3–E5 experiment result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h{m:02d}m{s:02d}s"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def _load_results(exp_glob: str, domain: str | None) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(EXPERIMENTS_DIR.glob(exp_glob)):
        if not path.is_file() or path.name.endswith("_harness.json"):
            continue
        payload = json.loads(path.read_text())
        if domain and payload.get("domain") != domain:
            continue
        summary = payload.get("summary") or {}
        if not summary:
            continue
        timing = payload.get("timing") or {}
        rows.append(
            {
                "path": path,
                "experiment": payload.get("experiment"),
                "run_id": path.stem,
                "domain": payload.get("domain"),
                "pass@k": summary.get("pass@k"),
                "timing": timing,
                "rounds": payload.get("rounds") or [],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize experiment step timings")
    parser.add_argument(
        "--experiment",
        choices=["E3", "E4", "E5", "all"],
        default="all",
    )
    parser.add_argument("--domain", choices=["airline", "retail"], default=None)
    args = parser.parse_args()

    globs = {
        "E3": "E3_original_evolution/**/*.json",
        "E4": "E4_cross_instance_evolution/**/*.json",
        "E5": "E5_debate/**/*.json",
    }
    selected = globs.keys() if args.experiment == "all" else [args.experiment]

    print("| Exp | Run ID | Domain | Total | Train | Final test | pass@k |")
    print("|-----|--------|--------|-------|-------|------------|--------|")
    for exp in selected:
        for row in _load_results(globs[exp], args.domain):
            timing = row["timing"]
            if not timing:
                continue
            print(
                f"| {row['experiment']} | {row['run_id']} | {row['domain']} | "
                f"{_fmt_seconds(timing.get('duration_seconds'))} | "
                f"{_fmt_seconds(timing.get('train_duration_seconds'))} | "
                f"{_fmt_seconds(timing.get('final_test_duration_seconds'))} | "
                f"{row['pass@k']} |"
            )

            for rnd in row["rounds"]:
                rnd_num = rnd.get("round")
                rnd_dur = rnd.get("round_duration_seconds")
                print(f"\n  Round {rnd_num} ({_fmt_seconds(rnd_dur)}):")
                print(
                    "  | Step | Started | Duration |"
                )
                print(
                    "  |------|---------|----------|"
                )
                for step in rnd.get("steps") or []:
                    print(
                        f"  | {step.get('step')} | {step.get('started_at', '-')} | "
                        f"{_fmt_seconds(step.get('duration_seconds'))} |"
                    )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
