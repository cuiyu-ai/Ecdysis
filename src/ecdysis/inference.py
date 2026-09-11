"""Task-agnostic execution of a retained harness."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

HarnessT = TypeVar("HarnessT")
TaskT = TypeVar("TaskT")
OutputT = TypeVar("OutputT")

type InferenceExecutor[HarnessT, TaskT, OutputT] = Callable[
    [HarnessT, TaskT], OutputT
]
type TaskId = Callable[[TaskT, int], str]
type OutputEncoder[OutputT] = Callable[[OutputT], Any]


def default_task_id(task: Any, index: int) -> str:
    """Use a supplied task identifier when available, otherwise the index."""
    if isinstance(task, dict):
        identifier = task.get("task_id", task.get("id"))
        if identifier is not None:
            return str(identifier)
    return str(index)


@dataclass(frozen=True)
class InferenceRecord[TaskT, OutputT]:
    """Outcome and timing information for one inference task."""

    index: int
    task_id: str
    task: TaskT
    output: OutputT | None
    elapsed_seconds: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self, output_encoder: OutputEncoder[OutputT] | None = None) -> dict:
        payload = asdict(self)
        if self.error is None and output_encoder is not None:
            payload["output"] = output_encoder(self.output)  # type: ignore[arg-type]
        return payload


@dataclass(frozen=True)
class InferenceResult[TaskT, OutputT]:
    """Stable batch result returned by :class:`InferenceRunner`."""

    records: tuple[InferenceRecord[TaskT, OutputT], ...]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def succeeded(self) -> int:
        return sum(record.succeeded for record in self.records)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded

    def to_dict(
        self, output_encoder: OutputEncoder[OutputT] | None = None
    ) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "records": [
                record.to_dict(output_encoder) for record in self.records
            ],
        }

    def write_jsonl(
        self,
        path: str | Path,
        *,
        output_encoder: OutputEncoder[OutputT] | None = None,
    ) -> Path:
        """Write one JSON object per task, creating parent directories."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(
                    json.dumps(
                        record.to_dict(output_encoder),
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return output_path


class InferenceRunner[HarnessT, TaskT, OutputT]:
    """Run a frozen harness through a user-supplied execution adapter."""

    def __init__(
        self,
        harness: HarnessT,
        executor: InferenceExecutor[HarnessT, TaskT, OutputT],
        *,
        task_id: TaskId[TaskT] = default_task_id,
    ) -> None:
        self.harness = harness
        self.executor = executor
        self.task_id = task_id

    def run(
        self,
        tasks: Iterable[TaskT],
        *,
        fail_fast: bool = False,
    ) -> InferenceResult[TaskT, OutputT]:
        """Execute tasks sequentially without adapting the retained harness."""
        records: list[InferenceRecord[TaskT, OutputT]] = []
        for index, task in enumerate(tasks):
            started = perf_counter()
            try:
                output = self.executor(self.harness, task)
            except Exception as exc:
                record = InferenceRecord(
                    index=index,
                    task_id=self.task_id(task, index),
                    task=task,
                    output=None,
                    elapsed_seconds=perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                records.append(record)
                if fail_fast:
                    raise
            else:
                records.append(
                    InferenceRecord(
                        index=index,
                        task_id=self.task_id(task, index),
                        task=task,
                        output=output,
                        elapsed_seconds=perf_counter() - started,
                    )
                )
        return InferenceResult(records=tuple(records))
