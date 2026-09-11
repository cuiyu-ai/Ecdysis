# Ecdysis

Official method-level implementation for
[*Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents*](https://arxiv.org/abs/2609.11677).

Ecdysis improves a runtime harness while keeping the task model, execution
environment, and scoring function fixed. The public package contains the general
training algorithm, FDCR, artifact schemas, and synthetic tests. Task-specific
assets and local run files are intentionally outside this repository.

## Method

Each training round follows the paper's acceptance loop:

1. Collect trajectories with the currently retained harness.
2. Mark a trajectory as failed when its fixed score is below `failure_threshold`.
3. Aggregate structured failure evidence and prioritize patterns recurring across
   distinct task instances; singleton patterns remain auxiliary evidence.
4. Run Failure-Driven Collaborative Refinement (FDCR) for `refinement_passes`.
5. Pass the structured specification to an isolated candidate editor.
6. Retain the candidate only when its training score strictly improves.

After the final round, the retained harness is frozen for inference.

## Paper To Code

| Paper component | Implementation |
|---|---|
| Failure signal and evidence aggregation | `ecdysis.evolution.collect_failed_trajectories` |
| Batch-level cross-instance grouping | `ecdysis.evolution.group_failures_by_pattern` |
| FDCR roles and moderator | `ecdysis.fdcr.run_harness_fdcr` |
| Candidate validation and freezing | `ecdysis.training.EcdysisTrainer` |
| Frozen-harness inference | `ecdysis.training.TrainingResult.infer` |
| Reusable learned behavior | `ecdysis.artifacts.SkillArtifact` |

## Install

Ecdysis requires Python 3.12 or 3.13.

```bash
pip install -e ".[dev]"
```

## Quickstart

Run the synthetic end-to-end training and batch-inference example:

```bash
python examples/quickstart.py
```

Expected output:

```text
final_score=1.0
example task @ revision 1
second task @ revision 1
```

To analyze an existing JSON result file without running FDCR:

```bash
ecdysis path/to/results.json
```

The input is a JSON object with `tasks` and `simulations` arrays. Each simulation
may contain `task_id`, `trial`, `messages`, `termination_reason`, and a
`reward_info` object with `reward` and `reward_breakdown`.

## Integration Contract

`EcdysisTrainer` has three external adapters:

- `collector(harness)` executes the fixed task model and environment, returning
  scored trajectory records.
- `editor(harness, specification, failures, groups)` creates an isolated candidate
  without mutating the retained harness.
- `scorer(results)` computes the fixed aggregate training score. The default is
  the arithmetic mean of available trajectory rewards.

This separation keeps FDCR responsible for diagnosis and specification, while the
editor remains responsible for implementation.

## Inference Contract

After training, `TrainingResult` keeps the accepted harness frozen. Use
`infer_many(tasks, executor)` or `InferenceRunner` to run a batch without further
adaptation. The executor is the only task or environment adapter:

```python
from ecdysis.inference import InferenceRunner

runner = InferenceRunner(
    frozen_harness,
    lambda harness, task: execute_with_environment(harness, task),
)
result = runner.run(tasks)
result.write_jsonl("inference.jsonl")
```

Each record contains the task, stable task identifier, output or error, and
elapsed time. Failed tasks are retained in the batch result; pass `fail_fast=True`
when the surrounding evaluation requires immediate interruption.

## External Benchmarks

The repository does not redistribute benchmark data. Use the same adapter
contract for the paper's external benchmarks, including tau2 and AgentBench:

- `load_tasks(split)` loads tasks from the user's separately installed benchmark;
- `execute(harness, task)` runs one task with the frozen harness;
- `encode_output(output)` converts the benchmark result to JSON-safe data;
- `name` and `version` record the external benchmark and pinned commit/version.

The generic runner is available as both Python API (`run_benchmark`) and CLI:

```bash
ecdysis-run \
  --adapter my_tau2_adapter:build_adapter \
  --harness my_harness:load_frozen_harness \
  --split test \
  --output results/inference.jsonl
```

The adapter module, benchmark installation, and harness are user-provided. No
dataset, secret, or local configuration is read from this repository. Record the
external benchmark commit and the command used for each reported experiment.

## Repository Layout

```text
src/ecdysis/
  evolution.py       Failure extraction, aggregation, and stability checks
  training.py        Training loop, strict validation, and frozen inference
  inference.py       Batch inference runner and JSONL result records
  benchmark.py       External benchmark adapter protocol and CLI runner
  artifacts.py       Versioned reusable-skill artifacts
  fdcr/              Collaborative refinement and moderator synthesis
  llm_client.py       Compatible chat-completions client
examples/
  quickstart.py      Synthetic end-to-end example
tests/                Unit and integration tests with synthetic records
```

## Development

```bash
python -m pytest
ruff check .
```

The public-tree test rejects local settings, generated outputs, legacy terminology,
and task-specific source code.
