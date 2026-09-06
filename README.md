# Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents

[Project Page](https://github.com/cuiyu-ai/Ecdysis) | **Project Lead:** Yu Cui (<cuiyu@bit.edu.cn>)

**Research status:** ongoing research release. This repository publishes the main Ecdysis implementation used for runtime-harness evolution while excluding local experiment configuration files, credentials, raw benchmark data, generated traces, and private run artifacts.

## Motivation

Existing harness-evolution workflows commonly inspect one failed trajectory at a time. That local view can produce narrow patches, duplicate effort across related failures, and miss failure modes that only become clear when several task instances are considered together.

Ecdysis asks a different question: **can shared failure structure across tasks guide more general, auditable runtime-harness updates?**

## Method

The workflow has four stages:

1. Run the current agent and harness on a batch of task instances.
2. Extract and group failures that share a recurring behavioral or policy pattern.
3. Analyze each group collectively and produce a concrete harness-update specification.
4. Apply the update in an isolated staging environment, then evaluate it on held-out tasks.

The research design studies two complementary components:

- **Mixed Training:** groups recurring failure patterns across task instances and reasons over them jointly instead of patching each trajectory independently.
- **Multi-Agent Debate (MAD):** uses analyst, critic, engineer, and moderator roles to challenge the diagnosis and synthesize an implementation-ready update specification.

The intended comparison progresses from no harness and a frozen harness, through serial single-task evolution, to Mixed Training and the full Mixed Training + MAD method.

## Experimental Results

Macro-averaged results across the ten model-domain cells:

| Method | AVG (%) | Pass@3 (%) | Pass^3 (%) | Tokens (M) | Time (s) |
|---|---:|---:|---:|---:|---:|
| Direct | 38.17 &plusmn; 5.85 | 53.50 | 22.50 | 8.701 &plusmn; 0.863 | 87.91 &plusmn; 10.06 |
| Human-Aug. | 51.67 &plusmn; 8.47 | 66.00 | 37.00 | 11.349 &plusmn; 1.298 | 118.53 &plusmn; 20.90 |
| Self-Evolution | 46.67 &plusmn; 9.22 | 63.50 | 29.00 | 11.567 &plusmn; 1.148 | 126.08 &plusmn; 16.59 |
| **Ecdysis (w/o MAD)** | 54.67 &plusmn; 5.55 | 66.50 | 42.00 | 10.354 &plusmn; 1.169 | 131.42 &plusmn; 24.29 |
| **Ecdysis (w/ MAD)** | **59.33 &plusmn; 5.08** | **71.50** | **45.00** | 10.157 &plusmn; 0.852 | 118.69 &plusmn; 13.27 |

Training time (s) for the three harness self-evolution methods:

| Dataset | Self-Evolution | Ecdysis (w/o MAD) | Ecdysis (w/ MAD) | Speedup |
|---|---:|---:|---:|---:|
| Retail | 1,831.4 | 1,292.4 | 1,405.9 | 1.42&times; / 1.30&times; |
| Airline | 8,120.6 | 2,510.8 | 4,403.0 | **3.23&times;** / 1.84&times; |

## Repository Layout

```text
src/ecdysis/
  evolution.py                 Failure processing and harness-evolution analysis
  artifacts.py                 Versioned evolved-skill artifacts
  experiment_config.py         Shared experiment defaults and command construction
  harness_patch.py             Patch application and artifact persistence
  harness_replay.py            Replay preflight for staged harness changes
  experiments/                 Experiment runner implementations
  pipeline/                    Shared evaluation, logging, timing, and guardrail steps
  mad/                         Multi-agent debate roles and synthesis
  harness/                     Extracted runtime-harness modules
scripts/                       Evaluation, experiment, aggregation, and timing CLIs
tests/                         Public-tree and core artifact tests
```

## Usage

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Run the lightweight checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

Experiment scripts accept user-supplied local YAML run files. Those files are intentionally not part of this public release.

## Acknowledgments

This project is motivated in part by [Life-Harness](https://arxiv.org/abs/2605.22166) and ongoing work on self-improving agent harnesses.

## Citation

```bibtex
@misc{ecdysis2026,
  title        = {Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents},
  author       = {Ruiqing Yue and Yu Cui and Xianhong Xue and Tingyu Li and Ting Li and Wenzhuo Zhu and Zhe Cui and Haibin Zhang and Cong Zuo},
  year         = {2026},
  howpublished = {\url{https://github.com/cuiyu-ai/Ecdysis}},
  note         = {Code repository}
}
```
