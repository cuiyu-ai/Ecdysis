"""External benchmark adapter protocol and reproducible inference entry point."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from ecdysis.inference import InferenceResult, InferenceRunner

TaskT = TypeVar("TaskT")
OutputT = TypeVar("OutputT")
HarnessT = TypeVar("HarnessT")


class BenchmarkAdapter(Protocol[TaskT, OutputT]):
    """Adapter supplied by an externally managed benchmark installation."""

    name: str
    version: str

    def load_tasks(self, split: str) -> Iterable[TaskT]: ...

    def execute(self, harness: Any, task: TaskT) -> OutputT: ...

    def encode_output(self, output: OutputT) -> Any: ...


@dataclass(frozen=True)
class BenchmarkRun:
    """Metadata and outcomes for one external benchmark inference run."""

    benchmark: str
    version: str
    split: str
    result: InferenceResult[Any, Any]

    def to_dict(
        self, *, encoder: Callable[[Any], Any] | None = None
    ) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "version": self.version,
            "split": self.split,
            **self.result.to_dict(encoder),
        }

    def write_jsonl(
        self, path: str | Path, *, encoder: Callable[[Any], Any]
    ) -> Path:
        """Write run records with benchmark metadata on every line."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.result.records:
                row = {
                    "benchmark": self.benchmark,
                    "version": self.version,
                    "split": self.split,
                    **record.to_dict(encoder),
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, default=str) + "\n"
                )
        return output_path


def run_benchmark[TaskT, OutputT, HarnessT](
    adapter: BenchmarkAdapter[TaskT, OutputT],
    harness: HarnessT,
    *,
    split: str,
    fail_fast: bool = False,
) -> BenchmarkRun:
    """Run an externally loaded benchmark through a frozen harness."""
    result = InferenceRunner(harness, adapter.execute).run(
        adapter.load_tasks(split),
        fail_fast=fail_fast,
    )
    return BenchmarkRun(
        benchmark=str(adapter.name),
        version=str(adapter.version),
        split=split,
        result=result,
    )


def _load_symbol(specification: str) -> Any:
    module_name, separator, symbol_name = specification.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError("expected import path in the form module:callable")
    return getattr(importlib.import_module(module_name), symbol_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a frozen harness through an external benchmark adapter."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Import path to an adapter factory: module:callable",
    )
    parser.add_argument(
        "--harness",
        required=True,
        help="Import path to a frozen-harness factory: module:callable",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    adapter = _load_symbol(args.adapter)()
    harness = _load_symbol(args.harness)()
    run = run_benchmark(
        adapter,
        harness,
        split=args.split,
        fail_fast=args.fail_fast,
    )
    run.write_jsonl(args.output, encoder=adapter.encode_output)
    print(
        json.dumps(
            run.to_dict(encoder=adapter.encode_output),
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
