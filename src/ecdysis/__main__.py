"""Command-line entry point for the public Ecdysis algorithms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecdysis.evolution import collect_failed_trajectories, group_failures_by_pattern


def _group_summary(groups: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    return {
        key: {
            "failures": len(members),
            "tasks": len({str(member.get("task_id")) for member in members}),
        }
        for key, members in sorted(groups.items())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract and group recurring failure evidence."
    )
    parser.add_argument("results", type=Path, help="JSON execution-results file")
    parser.add_argument("--max-messages", type=int, default=80)
    args = parser.parse_args(argv)

    failures = collect_failed_trajectories(
        args.results,
        max_messages=args.max_messages,
    )
    groups = group_failures_by_pattern(failures)
    print(
        json.dumps(
            {
                "failure_count": len(failures),
                "groups": _group_summary(groups),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
