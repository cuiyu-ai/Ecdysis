# Ecdysis package

`src/ecdysis/` contains the first-party implementation for runtime-harness evolution.

## Modules

| Path | Purpose |
|---|---|
| `evolution.py` | Failure extraction, mixed-training grouping, OpenCode staging, and evolution analysis helpers |
| `artifacts.py` | Versioned evolved-skill artifacts |
| `harness_patch.py` | Harness patch representation, application, and persistence |
| `harness_replay.py` | Replay preflight checks for staged harness changes |
| `experiment_config.py` | Shared experiment defaults and command-building helpers |
| `experiments/` | Experiment runner classes |
| `pipeline/` | Shared evaluation, logging, timing, persistence, and guardrail steps |
| `mad/` | Multi-agent debate roles and moderator synthesis |
| `harness/` | Extracted runtime-harness modules |

## Artifacts

Ecdysis stores learned harness behavior in two structured forms:

1. **H5 skills:** `SkillArtifact` JSON, loaded by evaluation with `--skill-artifact`.
2. **H2/H3/H4 patches:** `HarnessPatch` JSON, applied to the runtime harness between training rounds.

Generated artifacts, local run settings, raw traces, and secrets are intentionally excluded from the public tree.
