"""Multi-Agent Debate (MAD) for harness evolution.

Three roles debate how to update the harness from failure clusters;
a moderator synthesizes a patch spec that OpenCode applies in staging.
"""

from .debate import run_harness_mad, save_mad_transcript

__all__ = ["run_harness_mad", "save_mad_transcript"]
