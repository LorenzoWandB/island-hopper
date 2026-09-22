"""Core game rules. Pure Python, no I/O, deterministic.

Coordinates: x grows east (right), y grows south (down). Facing is one of N/E/S/W.
A level is a set of walkable tiles floating over water. Walking off a tile = splash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Dir = Literal["N", "E", "S", "W"]
Pos = tuple[int, int]

COMMANDS = ("move_forward", "turn_left", "turn_right", "collect_gem", "toggle_switch")

DIR_VEC: dict[str, Pos] = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT = {v: k for k, v in LEFT.items()}

# Scoring. Keep in one place so the RL reward and the eval scorer agree.
SCORE_GOAL = 50
SCORE_GEM = 20
SCORE_PER_COMMAND = -2
SCORE_FALL = -30
SCORE_EXTRA_ATTEMPT = -15
SCORE_WASTED_ACTION = -3  # collect_gem with no gem, toggle_switch with no switch
SCORE_INVALID = -40       # unknown command or over the length limit. Must be worse than falling, or RL learns to submit garbage.
SCORE_NO_GOAL = -20       # program ended (or ran out of commands) without reaching the portal. Otherwise an empty program scores 0.


@dataclass
class Switch:
    pos: Pos
    bridge: list[Pos]  # tiles that exist only while the switch is on


@dataclass
class Level:
    id: str
    name: str
    tiles: set[Pos]
    start: Pos
    facing: str
    goal: Pos
    gems: set[Pos] = field(default_factory=set)
    switches: list[Switch] = field(default_factory=list)
    max_commands: int = 20
    hint: str = ""

    def bounds(self) -> tuple[int, int, int, int]:
        pts = set(self.tiles) | {self.start, self.goal}
        for s in self.switches:
            pts |= set(s.bridge)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tiles": sorted(self.tiles),
            "start": list(self.start),
            "facing": self.facing,
            "goal": list(self.goal),
            "gems": sorted(self.gems),
            "switches": [{"pos": list(s.pos), "bridge": [list(b) for b in s.bridge]} for s in self.switches],
            "max_commands": self.max_commands,
            "hint": self.hint,
        }


@dataclass
class StepEvent:
    index: int
    command: str
    pos: Pos
    facing: str
    outcome: str  # moved | turned | collected | toggled | fell | wasted | goal | blocked
    note: str = ""
    gems_left: int = 0
    switches_on: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index, "command": self.command, "pos": list(self.pos), "facing": self.facing,
            "outcome": self.outcome, "note": self.note, "gems_left": self.gems_left, "switches_on": self.switches_on,
        }


@dataclass
class RunResult:
    level_id: str
    attempt: int
    status: str  # goal | fell | out_of_commands | ended | invalid
    steps: list[StepEvent]
    gems_collected: int
    gems_total: int
    commands_used: int
    score: int
    final_pos: Pos
    final_facing: str
    message: str

    def to_dict(self) -> dict:
        return {
            "level_id": self.level_id, "attempt": self.attempt, "status": self.status,
            "steps": [s.to_dict() for s in self.steps], "gems_collected": self.gems_collected,
            "gems_total": self.gems_total, "commands_used": self.commands_used, "score": self.score,
            "final_pos": list(self.final_pos), "final_facing": self.final_facing, "message": self.message,
        }


class Episode:
    """One agent's attempts at one level. Tracks attempt count for the extra-attempt penalty."""

    def __init__(self, level: Level, max_attempts: int = 3):
        self.level = level
        self.max_attempts = max_attempts
        self.attempts: list[RunResult] = []
        self.solved = False

    @property
    def attempts_left(self) -> int:
        return self.max_attempts - len(self.attempts)

    def run(self, commands: list[str]) -> RunResult:
        if self.solved:
            raise ValueError("level already solved; call reset")
        if self.attempts_left <= 0:
            raise ValueError("no attempts left; call reset")
        attempt = len(self.attempts) + 1
        result = run_program(self.level, commands, attempt=attempt)
        self.attempts.append(result)
        if result.status == "goal":
            self.solved = True
        return result

    def best_score(self) -> int:
        return max((a.score for a in self.attempts), default=0)

    def reset(self) -> None:
        self.attempts = []
        self.solved = False


def run_program(level: Level, commands: list[str], attempt: int = 1) -> RunResult:
    """Execute a whole program against a level. Never raises on bad commands; reports them."""
    bad = [c for c in commands if c not in COMMANDS]
    if bad:
        return RunResult(
            level.id, attempt, "invalid", [], 0, len(level.gems), 0,
            SCORE_INVALID + SCORE_EXTRA_ATTEMPT * (attempt - 1), level.start, level.facing,
            f"Unknown command(s): {bad}. Valid commands: {list(COMMANDS)}",
        )
    if len(commands) > level.max_commands:
        return RunResult(
            level.id, attempt, "invalid", [], 0, len(level.gems), 0,
            SCORE_INVALID + SCORE_EXTRA_ATTEMPT * (attempt - 1), level.start, level.facing,
            f"Program too long: {len(commands)} commands, max is {level.max_commands}.",
        )

    pos: Pos = level.start
    facing = level.facing
    gems = set(level.gems)
    switch_state = [False for _ in level.switches]
    steps: list[StepEvent] = []
    score = SCORE_EXTRA_ATTEMPT * (attempt - 1)
    status = "ended"
    message = ""

    def walkable(p: Pos) -> bool:
        if p in level.tiles:
            return True
        for on, sw in zip(switch_state, level.switches):
            if on and p in sw.bridge:
                return True
        return False

    for i, cmd in enumerate(commands):
        score += SCORE_PER_COMMAND
        outcome, note = "", ""
        if cmd == "turn_left":
            facing = LEFT[facing]
            outcome = "turned"
        elif cmd == "turn_right":
            facing = RIGHT[facing]
            outcome = "turned"
        elif cmd == "move_forward":
            dx, dy = DIR_VEC[facing]
            nxt = (pos[0] + dx, pos[1] + dy)
            if walkable(nxt):
                pos = nxt
                outcome = "moved"
            else:
                pos = nxt
                outcome = "fell"
                score += SCORE_FALL
                status = "fell"
                message = f"Line {i + 1}: move_forward walked off the island at {nxt}. Splash."
        elif cmd == "collect_gem":
            if pos in gems:
                gems.discard(pos)
                score += SCORE_GEM
                outcome = "collected"
            else:
                outcome = "wasted"
                score += SCORE_WASTED_ACTION
                note = "no gem here"
        elif cmd == "toggle_switch":
            hit = False
            for k, sw in enumerate(level.switches):
                if sw.pos == pos:
                    switch_state[k] = not switch_state[k]
                    hit = True
            if hit:
                outcome = "toggled"
            else:
                outcome = "wasted"
                score += SCORE_WASTED_ACTION
                note = "no switch here"

        steps.append(StepEvent(i, cmd, pos, facing, outcome, note, len(gems), list(switch_state)))

        if status == "fell":
            break
        if pos == level.goal and outcome == "moved":
            status = "goal"
            score += SCORE_GOAL
            steps[-1].outcome = "goal"
            missed = len(gems)
            message = "Reached the portal" + (f" with {missed} gem(s) left behind." if missed else " with every gem. Perfect.")
            break

    if status == "ended":
        score += SCORE_NO_GOAL
        message = f"Program finished at {pos} facing {facing} without reaching the portal at {level.goal}."
        if len(commands) >= level.max_commands:
            status = "out_of_commands"

    collected = len(level.gems) - len(gems)
    return RunResult(
        level.id, attempt, status, steps, collected, len(level.gems), len(commands),
        score, pos, facing, message,
    )


# ---------- Rendering for the LLM ----------

def ascii_map(level: Level, pos: Pos | None = None, facing: str | None = None,
              switches_on: list[bool] | None = None, gems: set[Pos] | None = None) -> str:
    x0, y0, x1, y1 = level.bounds()
    pos = pos or level.start
    facing = facing or level.facing
    gems = level.gems if gems is None else gems
    switches_on = switches_on or [False] * len(level.switches)
    arrow = {"N": "^", "E": ">", "S": "v", "W": "<"}[facing]
    rows = []
    header = "    " + " ".join(f"{x:>2}" for x in range(x0, x1 + 1))
    rows.append(header)
    for y in range(y0, y1 + 1):
        cells = []
        for x in range(x0, x1 + 1):
            p = (x, y)
            ch = " ."
            if p in level.tiles:
                ch = " #"
            for on, sw in zip(switches_on, level.switches):
                if p in sw.bridge:
                    ch = " =" if on else " ~"
            for sw in level.switches:
                if p == sw.pos:
                    ch = " S"
            if p == level.goal:
                ch = " O"
            if p in gems:
                ch = " G"
            if p == pos:
                ch = " " + arrow
            cells.append(ch)
        rows.append(f"{y:>2}  " + " ".join(c.strip().rjust(2) for c in cells))
    return "\n".join(rows)


def describe_level(level: Level, attempts_left: int | None = None) -> str:
    """The text the agent sees. Deliberately complete: the task is planning, not perception."""
    sw_lines = ""
    if level.switches:
        sw_lines = "\nSwitches (stand on the switch tile and call toggle_switch to raise its bridge):\n" + "\n".join(
            f"  switch at {list(s.pos)} raises bridge tiles {[list(b) for b in s.bridge]}" for s in level.switches
        )
    def what_is(p: Pos) -> str:
        if p in level.tiles or p == level.goal:
            return "portal" if p == level.goal else ("gem" if p in level.gems else "tile")
        for sw in level.switches:
            if p in sw.bridge:
                return "bridge(down)"
        return "WATER"

    sx, sy = level.start
    around = ", ".join(f"{d}={what_is((sx + dx, sy + dy))}" for d, (dx, dy) in DIR_VEC.items())
    lines = [
        f"LEVEL {level.id}: {level.name}",
        "",
        "Map (x grows to the right/east, y grows downward/south):",
        ascii_map(level),
        "",
        "Legend: # walkable tile, . water (fatal), G gem on a tile, O portal (goal), S switch tile,",
        "        ~ bridge tile that is DOWN (water until toggled), = bridge UP, ^ > v < you and the direction you face",
        "",
        f"You start at {list(level.start)} facing {level.facing}. Portal at {list(level.goal)}.",
        f"Next to your start tile: {around}. move_forward now would go {level.facing} onto {what_is((sx + DIR_VEC[level.facing][0], sy + DIR_VEC[level.facing][1]))}.",
        "Turning: facing N: turn_left->W, turn_right->E. facing E: turn_left->N, turn_right->S. "
        "facing S: turn_left->E, turn_right->W. facing W: turn_left->S, turn_right->N.",
        f"Gems at: {[list(g) for g in sorted(level.gems)] or 'none'}" + sw_lines,
        "",
        f"Commands: {', '.join(COMMANDS)}. Max {level.max_commands} commands per program.",
        "move_forward moves one tile in the facing direction. Stepping onto water ends the run with a splash.",
        "collect_gem only works while standing on a gem's tile. The run ends the moment you step onto the portal.",
        f"Scoring: +{SCORE_GOAL} portal, +{SCORE_GEM} per gem, {SCORE_PER_COMMAND} per command, {SCORE_FALL} for falling, "
        f"{SCORE_NO_GOAL} for finishing without reaching the portal, {SCORE_INVALID} for an invalid or over-long program, "
        f"{SCORE_WASTED_ACTION} for a wasted collect/toggle, {SCORE_EXTRA_ATTEMPT} per extra attempt.",
    ]
    if attempts_left is not None:
        lines.append(f"Attempts left on this level: {attempts_left}.")
    if level.hint:
        lines.append(f"Hint: {level.hint}")
    return "\n".join(lines)
