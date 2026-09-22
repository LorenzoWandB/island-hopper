"""MCP server exposing the game to an agent. Runs over stdio.

Tools: list_levels, get_level, run_program, reset_level, get_progress.
Every call also POSTs an event to the renderer (if it's running) so the browser animates the play.
"""
from __future__ import annotations

import os
from typing import Any


from mcp.server.mcpserver import MCPServer

from .engine import Episode, describe_level, COMMANDS
from .levels import CAMPAIGN, get_level as load_level
from .renderer_client import emit as _emit


MAX_ATTEMPTS = int(os.environ.get("ISLAND_MAX_ATTEMPTS", "3"))

mcp = MCPServer(
    "island-hopper",
    instructions=(
        "Island Hopper is a puzzle game. A small creature stands on floating tiles. You write a whole program "
        "(a list of commands) and run it. Call get_level to see the map, then run_program with your commands. "
        "If the run fails you get the exact step it failed on; you have a limited number of attempts per level."
    ),
)
_episodes: dict[str, Episode] = {}


def _episode(level_id: str) -> Episode:
    if level_id not in _episodes:
        _episodes[level_id] = Episode(load_level(level_id), max_attempts=MAX_ATTEMPTS)
    return _episodes[level_id]


@mcp.tool()
def list_levels() -> list[dict[str, Any]]:
    """List the campaign levels with id, name, gem count, and command budget."""
    return [
        {"id": lv.id, "name": lv.name, "gems": len(lv.gems), "switches": len(lv.switches),
         "max_commands": lv.max_commands}
        for lv in CAMPAIGN
    ]


@mcp.tool()
def get_level(level_id: str) -> str:
    """Show a level: ASCII map, legend, start position and facing, gem and portal coordinates, rules, and
    scoring. level_id is a campaign id ('1'..'10') or a procedural id like 'gen-3-42'."""
    ep = _episode(level_id)
    _emit({"type": "level", "level": ep.level.to_dict(), "attempt": len(ep.attempts) + 1, "source": "agent"})
    return describe_level(ep.level, attempts_left=ep.attempts_left)


@mcp.tool()
def run_program(level_id: str, commands: list[str], thought: str = "") -> dict[str, Any]:
    """Run a whole program on a level and get the result. commands is a list drawn from:
    move_forward, turn_left, turn_right, collect_gem, toggle_switch. thought is a one-line explanation of
    your plan; it is shown on screen and traced. Returns status (goal | fell | ended | out_of_commands | invalid),
    the step-by-step outcome, gems collected, score, and a message describing what went wrong, if anything."""
    ep = _episode(level_id)
    if ep.solved:
        return {"error": f"Level {level_id} is already solved. Call reset_level to replay it.",
                "best_score": ep.best_score()}
    if ep.attempts_left <= 0:
        return {"error": f"No attempts left on level {level_id}. Call reset_level to try again from scratch.",
                "best_score": ep.best_score()}
    attempt = len(ep.attempts) + 1
    _emit({"type": "level", "level": ep.level.to_dict(), "attempt": attempt, "source": "agent"})
    _emit({"type": "program", "commands": commands, "thought": thought, "source": "agent"})
    result = ep.run(commands)
    _emit({"type": "run", "result": result.to_dict(), "source": "agent"})
    out = result.to_dict()
    out["attempts_left"] = ep.attempts_left
    out["solved"] = ep.solved
    # Compact step log for the model; the full per-step list is still there.
    out["step_log"] = [
        f"{s.index + 1}. {s.command} -> {s.outcome}" + (f" ({s.note})" if s.note else "") + f" at {list(s.pos)} facing {s.facing}"
        for s in result.steps
    ]
    return out


@mcp.tool()
def reset_level(level_id: str) -> str:
    """Forget all attempts on a level and start fresh."""
    ep = _episode(level_id)
    ep.reset()
    _emit({"type": "level", "level": ep.level.to_dict(), "attempt": 1, "source": "agent"})
    return f"Level {level_id} reset. {ep.attempts_left} attempts available."


@mcp.tool()
def get_progress() -> dict[str, Any]:
    """Scoreboard: per level, whether it is solved, best score, and attempts used."""
    return {
        lid: {"solved": ep.solved, "best_score": ep.best_score(), "attempts_used": len(ep.attempts)}
        for lid, ep in _episodes.items()
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
