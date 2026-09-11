"""Failure-Driven Collaborative Refinement for harness evolution.

Three roles review how to update the harness from failure clusters;
a moderator synthesizes a patch spec that OpenCode applies in staging.
"""

from .review import format_update_spec, run_harness_fdcr, save_fdcr_transcript

__all__ = ["format_update_spec", "run_harness_fdcr", "save_fdcr_transcript"]
