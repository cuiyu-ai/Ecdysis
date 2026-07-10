# Scripts

These command-line tools support evaluation, experiment orchestration, aggregation, timing summaries, and harness replay validation.

| Script | Purpose |
|---|---|
| `run_experiment.py` | Dispatch an E1-E5 experiment from a user-supplied local YAML run file |
| `eval_harness.py` | Run one harness evaluation and write simulation summaries |
| `aggregate_results.py` | Aggregate experiment outputs into paper-table summaries |
| `summarize_train_timing.py` | Summarize training wall-clock intervals from run metadata |
| `validate_harness_replay.py` | Subprocess replay preflight for staged harness edits |
| `run_batch.py` | Batch launcher across models and experiment presets |
| `models.py` | Model list helpers for batch experiments |

The public repository intentionally omits local run settings, credentials, raw traces, and generated outputs. Full evaluation requires a compatible benchmark/runtime installation plus user-provided local settings.
