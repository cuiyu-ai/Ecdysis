"""Wall-clock timing for E3–E5 evolution pipeline steps."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from ecdysis.pipeline.log import log_step_timing


class PipelineTimer:
    """Records ISO timestamps and durations for each pipeline step."""

    def __init__(self) -> None:
        self.started_at = datetime.now().isoformat()
        self._run_start = time.monotonic()
        self.steps: list[dict[str, Any]] = []

    @contextmanager
    def step(
        self,
        name: str,
        *,
        label: str = "",
        round_num: int | None = None,
        phase: str = "train",
    ) -> Iterator[None]:
        started_at = datetime.now().isoformat()
        t0 = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - t0
            self.record(
                name,
                started_at=started_at,
                duration_seconds=duration,
                round_num=round_num,
                phase=phase,
            )
            if label:
                log_step_timing(label, round_num, name, duration)

    def record(
        self,
        name: str,
        *,
        started_at: str,
        duration_seconds: float,
        round_num: int | None = None,
        phase: str = "train",
        extra: dict[str, Any] | None = None,
    ) -> None:
        ended_at = datetime.now().isoformat()
        entry: dict[str, Any] = {
            "step": name,
            "phase": phase,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(duration_seconds, 3),
        }
        if round_num is not None:
            entry["round"] = round_num
        if extra:
            entry.update(extra)
        self.steps.append(entry)

    def steps_for_round(self, round_num: int) -> list[dict[str, Any]]:
        return [s for s in self.steps if s.get("round") == round_num]

    def round_duration(self, round_num: int) -> float:
        steps = self.steps_for_round(round_num)
        return round(sum(s["duration_seconds"] for s in steps), 3)

    def phase_duration(self, phase: str) -> float:
        total = sum(
            s["duration_seconds"] for s in self.steps if s.get("phase") == phase
        )
        return round(total, 3)

    def finish(self) -> dict[str, Any]:
        ended_at = datetime.now().isoformat()
        duration_seconds = round(time.monotonic() - self._run_start, 3)
        rounds: dict[str, Any] = {}
        for step in self.steps:
            rnd = step.get("round")
            if rnd is None:
                continue
            key = str(rnd)
            bucket = rounds.setdefault(
                key,
                {"round": rnd, "steps": [], "duration_seconds": 0.0},
            )
            bucket["steps"].append(step)
        for bucket in rounds.values():
            bucket["duration_seconds"] = round(
                sum(s["duration_seconds"] for s in bucket["steps"]),
                3,
            )
        return {
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "train_duration_seconds": self.phase_duration("train"),
            "final_test_duration_seconds": self.phase_duration("test"),
            "steps": self.steps,
            "rounds": rounds,
        }
