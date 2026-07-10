"""Prompt templates for evolution analysis, MAD roles, and prompt optimization.

Evolution prompts guide OpenCode harness edits. MAD role prompts live in
``ecdysis.mad.roles``; ``DEBATE_JUDGE_PROMPT`` is legacy (unused by E5).
"""

EVOLUTION_ANALYSIS_PROMPT = """You are a harness evolution expert. A harness is a set of runtime layers that help LLM agents avoid mistakes when interacting with deterministic environments.

## Harness Lifecycle Layers
The harness operates at four stages of the agent lifecycle:
- Environment Contract Layer (before interaction): Makes tool-use rules, policy constraints, and common pitfalls explicit in tool descriptions.
- Procedural Skill Layer (task conditioning): Retrieves reusable procedures from past trajectories and injects them into the system prompt.
- Action Realization Layer (before execution): Validates and canonicalizes model actions before they reach the environment, blocking deterministically-failing calls.
- Trajectory Regulation Layer (after execution): Monitors for repetition, stagnation, or budget exhaustion, and triggers recovery.

## Current Harness Code
```python
{harness_code}
```

## Failed Trajectories
{failed_trajectories}

## Your Task
Inspect the failed trajectories above and identify recurring failure patterns.
For each pattern:
1. Determine the earliest lifecycle point where it can be reliably detected or prevented.
2. Propose a targeted, minimal update to the appropriate harness layer.
3. Do NOT only return an analysis report. Implement the fix.

## Output Format
Return a JSON object with:
{{
  "skills": [
    {{
      "id": "snake_case_id",
      "title": "One-line summary",
      "pattern": "space-separated keywords for retrieval",
      "tip": "Actionable guidance (under 100 words)",
      "source_task_ids": ["task_id_1"],
      "issue_type": null
    }}
  ],
  "code_changes": [
    {{
      "file_path": "airline.py",
      "target": "h2_rule or h3_hint or h4_annotator",
      "description": "What this change does",
      "new_content": "COMPLETE new file content"
    }}
  ]
}}
"""


CROSS_INSTANCE_BATCH_ANALYSIS_PROMPT = """You are a harness evolution expert analyzing failures across MULTIPLE tasks to find common patterns.

## Harness Lifecycle Layers
The harness operates at four stages of the agent lifecycle:
- Environment Contract Layer (before interaction): Makes tool-use rules, policy constraints, and common pitfalls explicit in tool descriptions.
- Procedural Skill Layer (task conditioning): Retrieves reusable procedures from past trajectories and injects them into the system prompt.
- Action Realization Layer (before execution): Validates and canonicalizes model actions before they reach the environment, blocking deterministically-failing calls.
- Trajectory Regulation Layer (after execution): Monitors for repetition, stagnation, or budget exhaustion, and triggers recovery.

## Current Harness Code
```python
{harness_code}
```

## Failed Tasks (Grouped by Pattern)
{grouped_failures}

## Your Task
Instead of fixing each failure individually, find the UNDERLYING PATTERN that causes multiple tasks to fail.
For each pattern:
1. Determine the earliest lifecycle point where it can be reliably detected or prevented.
2. Propose a single targeted fix that closes the pattern for all affected tasks.
3. Check for regression: will this fix break any previously successful trajectory?

## Output Format
Return a JSON object with:
{{
  "skills": [
    {{
      "id": "snake_case_id",
      "title": "One-line summary",
      "pattern": "space-separated keywords for retrieval",
      "tip": "Actionable guidance (under 100 words)",
      "source_task_ids": ["task_id_1", "task_id_2"],
      "issue_type": null
    }}
  ],
  "code_changes": [
    {{
      "file_path": "airline.py",
      "target": "h2_rule or h3_hint or h4_annotator",
      "description": "What this change does and which tasks it fixes",
      "new_content": "COMPLETE new file content"
    }}
  ]
}}
"""


DEBATE_JUDGE_PROMPT = """You are a judge evaluating multiple execution trajectories for the same task.

## Task Description
{task_description}

## Trajectories
{trajectories}

## Evaluation Criteria
For each trajectory, score on these dimensions (0-10):
1. Correctness: Did the agent complete the task correctly?
2. Efficiency: Fewer steps is better.
3. Policy compliance: Did the agent follow the domain policy?
4. Communication: Was the interaction clear and helpful?

## Your Task
Select the BEST trajectory. Explain your reasoning.

## Output Format
Return a JSON object with:
{{
  "selected_trajectory": 0,
  "reasoning": "...",
  "scores": [
    {{
      "trajectory_index": 0,
      "correctness": 0-10,
      "efficiency": 0-10,
      "policy_compliance": 0-10,
      "communication": 0-10,
      "overall": 0-10
    }}
  ]
}}
"""
