# Package Map

| Module | Responsibility |
|---|---|
| `evolution.py` | Thresholded failure extraction, evidence compaction, cross-instance grouping, and score helpers |
| `training.py` | Collect-refine-edit-validate loop and frozen-harness inference |
| `inference.py` | Batch execution, outcome records, timing, and JSONL export |
| `benchmark.py` | External benchmark adapter protocol and CLI runner |
| `fdcr/` | Analyst, Critic, Engineer, and Moderator refinement passes |
| `artifacts.py` | Validation, merging, and persistence of reusable skills |
| `llm_client.py` | Minimal compatible chat-completions client |

The package uses dependency injection for task execution, candidate editing, and
scoring. Its core algorithms therefore remain independent of task-specific assets
and local run settings.
