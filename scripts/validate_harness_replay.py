#!/usr/bin/env python3
"""Subprocess entry: replay preflight for the live ``tau2/harness`` tree."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ecdysis.harness_replay import run_replay_validation  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_harness_replay.py <domain>", file=sys.stderr)
        return 2
    domain = sys.argv[1]
    failures = run_replay_validation(domain)
    if failures:
        print(json.dumps(failures, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
