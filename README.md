# Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents

> **Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents**

**Ruiqing Yue<sup>1,2</sup>**, **Yu Cui<sup>3</sup>**, **Xianhong Xue<sup>1,2</sup>**, **Tingyu Li<sup>3</sup>**, **Ting Li<sup>4</sup>**, **Zhe Cui<sup>1,2</sup>**, **Haibin Zhang<sup>5,6</sup>**, **Cong Zuo<sup>3</sup>**

<sup>1</sup> Chengdu Institute of Computer Applications, Chinese Academy of Sciences  
<sup>2</sup> University of Chinese Academy of Sciences  
<sup>3</sup> Beijing Institute of Technology  
<sup>4</sup> Beijing University of Technology  
<sup>5</sup> Yangtze Delta Region Institute of Tsinghua University, Zhejiang  
<sup>6</sup> Jiaxing Key Laboratory of Artificial Intelligence and Cyber Resilience

Ruiqing Yue and Yu Cui contributed equally to this work. Yu Cui proposed the algorithm and Ruiqing Yue performed the experiments.

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

- **Cross-Instance Learning:** groups recurring failure patterns across task instances and reasons over them jointly instead of patching each trajectory independently.
- **Multi-Agent Debate (MAD):** uses analyst, critic, engineer, and moderator roles to challenge the diagnosis and synthesize an implementation-ready update specification.

The intended comparison progresses from no harness and a frozen harness, through serial single-task evolution, to Cross-Instance Learning and the full Cross-Instance Learning + MAD method.

## Experimental Results

The current results use the `airline` test split with 20 tasks and 3 trials per task (60 simulations per configuration). The user simulator is `openai/deepseek-v4-flash` through DashScope-compatible mode. `pass@3` is the proportion of tasks with at least one successful trial out of three.

| Agent model | E1: No harness | E2: Frozen harness | E3: Serial evolution | E5: Cross-Instance Learning + MAD |
|---|:---:|:---:|:---:|:---:|
| qwen3-8b | 0.35 | 0.65 | 0.50 | 0.65 |
| qwen3-14b | 0.30 | 0.55 | 0.40 | 0.65 |
| qwen3-32b | 0.35 | 0.70 | **0.80** | **0.80** |

| Configuration | pass@3 | pass<sup>3</sup> | Average reward | Total tokens | Wall-clock time |
|---|:---:|:---:|:---:|:---:|:---:|
| qwen3-32b + E5 evolved harness | **0.80** | **0.35** | **0.583** | 12.1M | 12.0 min |

For qwen3-8b, the reported E3 run completes three training rounds and the reported E5 run completes two training rounds before the final test. For qwen3-14b and qwen3-32b, the evolved E3/E5 harnesses trained with qwen3-8b are reused and evaluated with the target agent model. `pass<sup>3</sup>` denotes the proportion of tasks successful in all three trials.

## qwen3-8b Training Time

Training time is measured as the wall-clock interval from the start of an evolution run to creation of its final evolved-harness artifact. It includes training evaluation, failure analysis, harness synthesis, and staging updates; E5 also includes the multi-agent debate. It excludes the final test.

| Method | Completed training rounds | Started | Finished | Training time |
|---|:---:|:---:|:---:|:---:|
| E3: Serial evolution | 3 | 2026-07-01 01:00:47 | 2026-07-01 04:50:42 | 3h 49m 56s |
| E4: Cross-Instance Learning | 3 | 2026-07-01 09:51:34 | 2026-07-01 10:30:29 | 38m 55s |
| E5: Cross-Instance Learning + MAD | 2 | 2026-07-01 23:50:29 | 2026-07-02 00:58:35 | 1h 08m 07s |

## Code Release

This release includes the Ecdysis first-party implementation:

- failure extraction, trajectory compaction, cross-instance grouping, and evolved-skill artifacts;
- E1-E5 experiment orchestration and shared pipeline utilities;
- multi-agent debate prompts and transcript handling;
- harness patching, replay preflight checks, and extracted runtime-harness modules;
- evaluation, aggregation, timing, and replay-validation scripts.

The release omits local YAML experiment files, secrets, generated output directories, raw traces, large benchmark data, and the full external benchmark framework. Full end-to-end evaluation expects a compatible benchmark/runtime installation and user-provided local run settings.

## Repository Layout

```text
src/ecdysis/
  evolution.py                 Failure processing and harness-evolution analysis
  artifacts.py                 Versioned evolved-skill artifacts
  experiment_config.py         Shared experiment defaults and command construction
  harness_patch.py             Patch application and artifact persistence
  harness_replay.py            Replay preflight for staged harness changes
  experiments/                 E1-E5 experiment runners
  pipeline/                    Shared evaluation, logging, timing, and guardrail steps
  mad/                         Multi-agent debate roles and synthesis
  harness/                     Extracted runtime-harness modules
scripts/                       Evaluation, experiment, aggregation, and timing CLIs
tests/                         Public-tree and core artifact tests
```

## Usage

Install the package in editable mode:

```bash
pip install -e .
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
  author       = {Ruiqing Yue and Yu Cui and Xianhong Xue and Tingyu Li and Zhe Cui and Haibin Zhang and Cong Zuo},
  year         = {2026},
  howpublished = {\url{https://github.com/cuiyu-ai/Ecdysis}},
  note         = {Code repository}
}
```
