"""Island Hopper: a tiny 3D puzzle game an LLM agent learns to play."""
from . import _env  # noqa: F401  loads .env before anything reads WANDB_* settings
from .engine import Level, Episode, StepEvent, RunResult, COMMANDS, describe_level
from .levels import CAMPAIGN, get_level, generate_level

__all__ = [
    "Level", "Episode", "StepEvent", "RunResult", "COMMANDS", "describe_level",
    "CAMPAIGN", "get_level", "generate_level",
]
