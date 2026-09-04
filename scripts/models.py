#!/usr/bin/env python3
"""Model list for Ecdysis experiments.

Qwen series + DeepSeek distilled models (4B-72B) for generalization experiments.
"""

# Qwen3 series (latest generation)
QWEN3_MODELS = [
    {"name": "Qwen3-4B", "id": "openai/qwen3-4b", "size": "4B", "family": "Qwen3"},
    {"name": "Qwen3-8B", "id": "openai/qwen3-8b", "size": "8B", "family": "Qwen3"},
    {"name": "Qwen3-14B", "id": "openai/qwen3-14b", "size": "14B", "family": "Qwen3"},
    {"name": "Qwen3-32B", "id": "openai/qwen3-32b", "size": "32B", "family": "Qwen3"},
]

# Qwen2.5 series (previous generation)
QWEN25_MODELS = [
    {"name": "Qwen2.5-7B", "id": "openai/qwen2.5-7b-instruct", "size": "7B", "family": "Qwen2.5"},
    {"name": "Qwen2.5-14B", "id": "openai/qwen2.5-14b-instruct", "size": "14B", "family": "Qwen2.5"},
    {"name": "Qwen2.5-32B", "id": "openai/qwen2.5-32b-instruct", "size": "32B", "family": "Qwen2.5"},
    {"name": "Qwen2.5-72B", "id": "openai/qwen2.5-72b-instruct", "size": "72B", "family": "Qwen2.5"},
]

# DeepSeek R1 distilled models (based on Qwen architecture)
DEEPSEEK_MODELS = [
    {"name": "DeepSeek-R1-Distill-Qwen-7B", "id": "openai/deepseek-r1-distill-qwen-7b", "size": "7B", "family": "DeepSeek"},
    {"name": "DeepSeek-R1-Distill-Qwen-14B", "id": "openai/deepseek-r1-distill-qwen-14b", "size": "14B", "family": "DeepSeek"},
    {"name": "DeepSeek-R1-Distill-Qwen-32B", "id": "openai/deepseek-r1-distill-qwen-32b", "size": "32B", "family": "DeepSeek"},
]

# All models combined
ALL_MODELS = QWEN3_MODELS + QWEN25_MODELS + DEEPSEEK_MODELS

# Model groups for different experiments
MODEL_GROUPS = {
    "qwen3": QWEN3_MODELS,
    "qwen2.5": QWEN25_MODELS,
    "deepseek": DEEPSEEK_MODELS,
    "all": ALL_MODELS,
    "small": [m for m in ALL_MODELS if float(m["size"].rstrip("B")) <= 8],
    "medium": [m for m in ALL_MODELS if 8 < float(m["size"].rstrip("B")) <= 32],
    "large": [m for m in ALL_MODELS if float(m["size"].rstrip("B")) > 32],
    # Representative subset for quick testing
    "core": [
        {"name": "Qwen3-8B", "id": "openai/qwen3-8b", "size": "8B", "family": "Qwen3"},
        {"name": "Qwen3-14B", "id": "openai/qwen3-14b", "size": "14B", "family": "Qwen3"},
        {"name": "Qwen2.5-7B", "id": "openai/qwen2.5-7b-instruct", "size": "7B", "family": "Qwen2.5"},
        {"name": "DeepSeek-R1-Distill-Qwen-7B", "id": "openai/deepseek-r1-distill-qwen-7b", "size": "7B", "family": "DeepSeek"},
    ],
}

# User simulator (fixed for all experiments)
USER_MODEL = "openai/deepseek-v4-flash"

# Evolution LLM (MAD for E5; used by run_batch only for E1/E2 defaults)
EVOLUTION_MODEL = "openai/qwen-max"
JUDGE_MODEL = "openai/qwen-plus"  # legacy; E5 uses MAD, not trajectory judge


def get_model_list(group: str = "all") -> list[dict]:
    """Get list of models for a given group."""
    return MODEL_GROUPS.get(group, ALL_MODELS)


def get_model_ids(group: str = "all") -> list[str]:
    """Get list of model IDs for a given group."""
    return [m["id"] for m in get_model_list(group)]


def get_model_names(group: str = "all") -> list[str]:
    """Get list of model names for a given group."""
    return [m["name"] for m in get_model_list(group)]


def print_model_table():
    """Print a formatted table of all models."""
    print(f"\n{'='*70}")
    print("  Available Models")
    print(f"{'='*70}")
    print(f"  {'Name':<35} {'Size':>6} {'Family':<10}")
    print(f"  {'-'*35} {'-'*6} {'-'*10}")
    for m in ALL_MODELS:
        print(f"  {m['name']:<35} {m['size']:>6} {m['family']:<10}")
    print(f"\n  Total: {len(ALL_MODELS)} models")
    print(f"{'='*70}\n")
